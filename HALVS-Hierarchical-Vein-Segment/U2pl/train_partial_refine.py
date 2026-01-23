import argparse
import copy
import logging
import os
import os.path as osp
import pprint
import random
import time
from datetime import datetime
from torch.optim import SGD
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.nn.functional as F
import yaml
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from tensorboardX import SummaryWriter
from u2pl.models.unet_de import UNet_LDMV2
from u2pl.dataset.augmentation import generate_unsup_data, generate_aug_data
from u2pl.dataset.builder import get_loader
from u2pl.models.model_helper import ModelBuilder
from u2pl.utils.dist_helper import setup_distributed
from u2pl.utils.loss_helper import (
    DiceLoss,
    DiceLoss_refine,
    compute_contra_memobank_loss,
    compute_unsupervised_loss,
    get_criterion,
    exclusive_loss
)
from u2pl.utils.lr_helper import get_optimizer, get_scheduler
from u2pl.utils.utils import (
    AverageMeter,
    get_rank,
    get_world_size,
    init_log,
    intersectionAndUnion,
    label_onehot,
    load_state,
    set_random_seed,
)

parser = argparse.ArgumentParser(description="Semi-Supervised Semantic Segmentation")
parser.add_argument("--config", type=str, default="config.yaml")
parser.add_argument("--exp_path", type=str, default=None)
parser.add_argument("--local_rank", type=int, default=0)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--port", default=None, type=int)


def main():
    global args, cfg, prototype
    args = parser.parse_args()
    seed = args.seed
    cfg = yaml.load(open(args.config, "r"), Loader=yaml.Loader)

    logger = init_log("global", logging.INFO)
    logger.propagate = 0

    cfg["exp_path"] = args.exp_path
    cfg["save_path"] = os.path.join(cfg["exp_path"], cfg["saver"]["snapshot_dir"])

    rank, word_size = setup_distributed(port=args.port)

    if rank == 0:
        if not osp.exists(cfg["exp_path"]):
            os.makedirs(cfg["exp_path"])
        if not osp.exists(cfg["save_path"]):
            os.makedirs(cfg["save_path"])
        logger.info("{}".format(pprint.pformat(cfg)))
        current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        tb_logger = SummaryWriter(
            cfg["exp_path"]
        )
    else:
        tb_logger = None

    cudnn.enabled = True
    cudnn.benchmark = True

    if args.seed is not None:
        print("set random seed to", args.seed)
        set_random_seed(args.seed)


    # Create network
    model = ModelBuilder(cfg["net"])
    modules_back = [model.encoder]
    if cfg["net"].get("aux_loss", False):
        modules_head = [model.auxor, model.decoder]
    else:
        modules_head = [model.decoder]

    if cfg["net"].get("sync_bn", True):
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    model.cuda()

    sup_loss_fn = get_criterion(cfg)

    # using partial labels
    using_partial = cfg['dataset'].get("using_partial", False)
    # train_loader_sup, train_loader_unsup, val_loader = get_loader(cfg, seed=seed)
    loaders = get_loader(cfg, seed=seed)
    if using_partial:
        train_loader_sup, train_loader_unsup, train_loader_partial, val_loader = loaders
    else:
        train_loader_sup, train_loader_unsup, val_loader = loaders
        train_loader_partial = None

    # Optimizer and lr decay scheduler
    cfg_trainer = cfg["trainer"]
    cfg_optim = cfg_trainer["optimizer"]
    times = 10 if "pascal" in cfg["dataset"]["type"] else 1

    params_list = []
    for module in modules_back:
        params_list.append(
            dict(params=module.parameters(), lr=cfg_optim["kwargs"]["lr"])
        )
    for module in modules_head:
        params_list.append(
            dict(params=module.parameters(), lr=cfg_optim["kwargs"]["lr"] * times)
        )

    optimizer = get_optimizer(params_list, cfg_optim)

    local_rank = int(os.environ["LOCAL_RANK"])
    model = torch.nn.parallel.DistributedDataParallel(
        model,
        device_ids=[local_rank],
        output_device=local_rank,
        find_unused_parameters=True,
        broadcast_buffers=False,
    )

    # Teacher model
    model_teacher = ModelBuilder(cfg["net"])
    model_teacher = model_teacher.cuda()
    model_teacher = torch.nn.parallel.DistributedDataParallel(
        model_teacher,
        device_ids=[local_rank],
        output_device=local_rank,
        find_unused_parameters=True,
    )

    for p in model_teacher.parameters():
        p.requires_grad = False

    best_prec = 0
    best_iou_3 = 0
    last_epoch = 0

    # auto_resume > pretrain
    if cfg["saver"].get("auto_resume", False):
        lastest_model = os.path.join(cfg["save_path"], "ckpt.pth")
        if not os.path.exists(lastest_model):
            "No checkpoint found in '{}'".format(lastest_model)
        else:
            print(f"Resume model from: '{lastest_model}'")
            best_prec, last_epoch = load_state(
                lastest_model, model, optimizer=optimizer, key="model_state"
            )
            _, _ = load_state(
                lastest_model, model_teacher, optimizer=optimizer, key="teacher_state"
            )

    elif cfg["saver"].get("pretrain", False):
        load_state(cfg["saver"]["pretrain"], model, key="model_state")
        load_state(cfg["saver"]["pretrain"], model_teacher, key="teacher_state")

    optimizer_start = get_optimizer(params_list, cfg_optim)
    lr_scheduler = get_scheduler(
        cfg_trainer, len(train_loader_sup), optimizer_start, start_epoch=last_epoch
    )

    # build class-wise memory bank
    memobank = []
    queue_ptrlis = []
    queue_size = []
    for i in range(cfg["net"]["num_classes"]):
        memobank.append([torch.zeros(0, 256)])
        queue_size.append(30000)
        queue_ptrlis.append(torch.zeros(1, dtype=torch.long))
    queue_size[0] = 50000

    # build prototype
    prototype = torch.zeros(
        (
            cfg["net"]["num_classes"],
            cfg["trainer"]["contrastive"]["num_queries"],
            1,
            256,
        )
    ).cuda()

    # Start to train model
    for epoch in range(last_epoch, cfg_trainer["epochs"]):
        # Training      
        train(
            model,
            model_teacher,
            optimizer,
            lr_scheduler,
            sup_loss_fn,
            train_loader_sup,
            train_loader_unsup,
            train_loader_partial,
            epoch,
            tb_logger,
            logger,
            memobank,
            queue_ptrlis,
            queue_size,
            using_partial
        )

        # Validation
        if cfg_trainer["eval_on"]:
            if rank == 0:
                logger.info("start evaluation")

            if epoch < cfg["trainer"].get("sup_only_epoch", 1):
                prec, iou_3 = validate(model, val_loader, epoch, logger)
            else:
                prec, iou_3 = validate(model_teacher, val_loader, epoch, logger)

            if rank == 0:
                state = {
                    "epoch": epoch + 1,
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "teacher_state": model_teacher.state_dict(),
                    "best_miou_without_bg": best_prec,
                }

                if iou_3 > best_iou_3:
                    best_iou_3 = iou_3

                if prec > best_prec:
                    best_prec = prec
                    state["best_miou_without_bg"] = prec
                    torch.save(
                        state, osp.join(cfg["save_path"], "ckpt_best.pth")
                    )

                torch.save(state, osp.join(cfg["save_path"], "ckpt.pth"))

                logger.info(
                    "\033[31m * Currently, the best val result is: {:.2f}\033[0m".format(
                        best_prec * 100
                    )
                )

                logger.info(
                    "\033[31m * Currently, the best 3rd IoU is: {:.2f}\033[0m".format(
                        best_iou_3 * 100
                    )
                )

                tb_logger.add_scalar("mIoU_without_bg val", prec, epoch)
            dist.barrier()
               
            


def train(
        model,
        model_teacher,
        optimizer,
        lr_scheduler,
        sup_loss_fn,
        loader_l,
        loader_u,
        loader_p,
        epoch,
        tb_logger,
        logger,
        memobank,
        queue_ptrlis,
        queue_size,
        using_partial=False
):
    global prototype
    ema_decay_origin = cfg["net"]["ema_decay"]

    model.train()

    loader_l.sampler.set_epoch(epoch)
    loader_u.sampler.set_epoch(epoch)
    loader_p.sampler.set_epoch(epoch)

    loader_l_iter = iter(loader_l)
    loader_u_iter = iter(loader_u)
    loader_p_iter = iter(loader_p)
    
    refine_model = UNet_LDMV2(in_chns=3+3, class_num=4, out_chns=4, ldm_method='replace', ldm_beta_sch='cosine', ts=10, ts_sample=2).cuda()
    criterion_dice = DiceLoss(n_classes=4)
    dice_refine = DiceLoss_refine(n_classes=4)
    local_rank = int(os.environ["LOCAL_RANK"])
    refine_optimizer = SGD(refine_model.parameters(), lr=0.001, momentum=0.9, weight_decay=0.0001)
    color_map = {
            0: (0, 0, 0),    # background class
            1: (255, 0, 0),  # class 1 (red)
            2: (0, 255, 0),  # class 2 (green)
            3: (0, 0, 255),  # class 3 (blue)
        }
    # criterion_l = nn.CrossEntropyLoss(**cfg['criterion']['kwargs']).cuda(local_rank)
    criterion_refine_ce = nn.CrossEntropyLoss()
    assert len(loader_l) == len(
        loader_u
    ), f"labeled data {len(loader_l)} unlabeled data {len(loader_u)}, imbalance!"
    assert len(loader_p) == len(
        loader_u
    ), f"labeled data {len(loader_l)} unlabeled data {len(loader_u)}, imbalance!"

    rank, world_size = dist.get_rank(), dist.get_world_size()

    sup_losses = AverageMeter(10)
    uns_losses = AverageMeter(10)
    con_losses = AverageMeter(10)
    # partial loss
    partial_losses = AverageMeter(10)
    partial_sup_losses = AverageMeter(10)
    partial_unsup_losses_1 = AverageMeter(10)
    partial_unsup_losses_2 = AverageMeter(10)
    
    data_times = AverageMeter(10)
    batch_times = AverageMeter(10)
    learning_rates = AverageMeter(10)

    batch_end = time.time()
    for step in range(len(loader_l)):
        batch_start = time.time()
        data_times.update(batch_start - batch_end)

        i_iter = epoch * len(loader_l) + step
        lr = lr_scheduler.get_lr()
        learning_rates.update(lr[0])
        lr_scheduler.step()

        image_l, label_l = loader_l_iter.next()
        #image_l, label_l = next(loader_l_iter)
        batch_size, h, w = label_l.size()
        image_l, label_l = image_l.cuda(), label_l.cuda()

        image_u, _ = loader_u_iter.next()
        # image_u,_ = next(loader_u_iter)
        image_u = image_u.cuda()

        # partial image and label
        image_p, label_p = loader_p_iter.next()
        # image_p,label_p = next(loader_p_iter)
        image_p, label_p = image_p.cuda(), label_p.cuda()

        if epoch < cfg["trainer"].get("sup_only_epoch", 1):
            contra_flag = "none"
            # forward
            outs = model(image_l)
            pred, rep = outs["pred"], outs["rep"]
            pred = F.interpolate(pred, (h, w), mode="bilinear", align_corners=True)

            # supervised loss
            if "aux_loss" in cfg["net"].keys():
                aux = outs["aux"]
                aux = F.interpolate(aux, (h, w), mode="bilinear", align_corners=True)
                sup_loss = sup_loss_fn([pred, aux], label_l)
            else:
                sup_loss = sup_loss_fn(pred, label_l)

            model_teacher.train()
            _ = model_teacher(image_l)

            unsup_loss = 0 * rep.sum()
            contra_loss = 0 * rep.sum()
            partial_loss = 0 * rep.sum()
            partial_sup_loss = 0 * rep.sum()
            partial_unsup_loss_1 = 0 * rep.sum()
            partial_unsup_loss_2 = 0 * rep.sum()
            
        else:
            if epoch == cfg["trainer"].get("sup_only_epoch", 1):
                # copy student parameters to teacher
                with torch.no_grad():
                    for t_params, s_params in zip(
                            model_teacher.parameters(), model.parameters()
                    ):
                        t_params.data = s_params.data


            # generate pseudo labels first
            model_teacher.eval()
            pred_u_teacher = model_teacher(image_u)["pred"]
            pred_u_teacher = F.interpolate(
                pred_u_teacher, (h, w), mode="bilinear", align_corners=True
            )
            pred_u_teacher = F.softmax(pred_u_teacher, dim=1)
            logits_u_aug, label_u_aug = torch.max(pred_u_teacher, dim=1)

            # apply strong data augmentation: cutout, cutmix, or classmix
            if np.random.uniform(0, 1) < 0.5 and cfg["trainer"]["unsupervised"].get(
                    "apply_aug", False
            ):
                image_u_aug, label_u_aug, logits_u_aug = generate_unsup_data(
                    image_u,
                    label_u_aug.clone(),
                    logits_u_aug.clone(),
                    mode=cfg["trainer"]["unsupervised"]["apply_aug"],
                )
            else:
                image_u_aug = image_u

            # forward
            num_labeled = len(image_l)
            image_all = torch.cat((image_l, image_u_aug))
            
             # get the strong augmented image
            image_p_aug = generate_aug_data(image_p)
            
            outs = model(image_all)
            
            # image_p is the weak augmented image

            # get the predictions of both weak augmented image and strong augmented image
            # pred_p_w, pred_p_s = model(image_p_all)["pred"].chunk(2)
            
            pred_p_s = model(image_p_aug)["pred"]
            
            pred_all, rep_all = outs["pred"], outs["rep"]
            pred_l, pred_u = pred_all[:num_labeled], pred_all[num_labeled:]
            
            pred_l_large = F.interpolate(
                pred_l, size=(h, w), mode="bilinear", align_corners=True
            )
            pred_u_large = F.interpolate(
                pred_u, size=(h, w), mode="bilinear", align_corners=True
            )
             
            pred_p_s = F.interpolate(
                pred_p_s, size=(h, w), mode="bilinear", align_corners=True
            )
            
            

            # supervised loss
            if "aux_loss" in cfg["net"].keys():
                aux = outs["aux"][:num_labeled]
                aux = F.interpolate(aux, (h, w), mode="bilinear", align_corners=True)
                sup_loss = sup_loss_fn([pred_l_large, aux], label_l.clone())
            else:
                sup_loss = sup_loss_fn(pred_l_large, label_l.clone())

            # teacher forward
            model_teacher.train()
            with torch.no_grad():
                out_t = model_teacher(image_all)
                pred_all_teacher, rep_all_teacher = out_t["pred"], out_t["rep"]
                prob_all_teacher = F.softmax(pred_all_teacher, dim=1)
                prob_l_teacher, prob_u_teacher = (
                    prob_all_teacher[:num_labeled],
                    prob_all_teacher[num_labeled:],
                )

                pred_u_teacher = pred_all_teacher[num_labeled:]
                pred_u_large_teacher = F.interpolate(
                    pred_u_teacher, size=(h, w), mode="bilinear", align_corners=True
                )
                

            # unsupervised loss
            drop_percent = cfg["trainer"]["unsupervised"].get("drop_percent", 100)
            percent_unreliable = (100 - drop_percent) * (1 - epoch / cfg["trainer"]["epochs"])
            drop_percent = 100 - percent_unreliable
            unsup_loss = (
                    compute_unsupervised_loss(
                        pred_u_large,
                        label_u_aug.clone(),
                        drop_percent,
                        pred_u_large_teacher.detach(),
                    )
                    * cfg["trainer"]["unsupervised"].get("loss_weight", 1)
            )

            # contrastive loss using unreliable pseudo labels
            contra_flag = "none"
            if cfg["trainer"].get("contrastive", False):
                cfg_contra = cfg["trainer"]["contrastive"]
                contra_flag = "{}:{}".format(
                    cfg_contra["low_rank"], cfg_contra["high_rank"]
                )
                alpha_t = cfg_contra["low_entropy_threshold"] * (
                        1 - epoch / cfg["trainer"]["epochs"]
                )

                with torch.no_grad():
                    prob = torch.softmax(pred_u_large_teacher, dim=1)
                    entropy = -torch.sum(prob * torch.log(prob + 1e-10), dim=1)

                    low_thresh = np.percentile(
                        entropy[label_u_aug != 255].cpu().numpy().flatten(), alpha_t
                    )
                    low_entropy_mask = (
                            entropy.le(low_thresh).float() * (label_u_aug != 255).bool()
                    )

                    high_thresh = np.percentile(
                        entropy[label_u_aug != 255].cpu().numpy().flatten(),
                        100 - alpha_t,
                    )
                    high_entropy_mask = (
                            entropy.ge(high_thresh).float() * (label_u_aug != 255).bool()
                    )

                    low_mask_all = torch.cat(
                        (
                            (label_l.unsqueeze(1) != 255).float(),
                            low_entropy_mask.unsqueeze(1),
                        )
                    )

                    low_mask_all = F.interpolate(
                        low_mask_all, size=pred_all.shape[2:], mode="nearest"
                    )
                    # down sample

                    if cfg_contra.get("negative_high_entropy", True):
                        contra_flag += " high"
                        high_mask_all = torch.cat(
                            (
                                (label_l.unsqueeze(1) != 255).float(),
                                high_entropy_mask.unsqueeze(1),
                            )
                        )
                    else:
                        contra_flag += " low"
                        high_mask_all = torch.cat(
                            (
                                (label_l.unsqueeze(1) != 255).float(),
                                torch.ones(logits_u_aug.shape)
                                    .float()
                                    .unsqueeze(1)
                                    .cuda(),
                            ),
                        )
                    high_mask_all = F.interpolate(
                        high_mask_all, size=pred_all.shape[2:], mode="nearest"
                    )  # down sample

                    # down sample and concat
                    label_l_small = F.interpolate(
                        label_onehot(label_l, cfg["net"]["num_classes"]),
                        size=pred_all.shape[2:],
                        mode="nearest",
                    )
                    label_u_small = F.interpolate(
                        label_onehot(label_u_aug, cfg["net"]["num_classes"]),
                        size=pred_all.shape[2:],
                        mode="nearest",
                    )

                if cfg_contra.get("binary", False):
                    contra_flag += " BCE"
                    contra_loss = compute_binary_memobank_loss(
                        rep_all,
                        torch.cat((label_l_small, label_u_small)).long(),
                        low_mask_all,
                        high_mask_all,
                        prob_all_teacher.detach(),
                        cfg_contra,
                        memobank,
                        queue_ptrlis,
                        queue_size,
                        rep_all_teacher.detach(),
                    )
                else:
                    if not cfg_contra.get("anchor_ema", False):
                        new_keys, contra_loss = compute_contra_memobank_loss(
                            rep_all,
                            label_l_small.long(),
                            label_u_small.long(),
                            prob_l_teacher.detach(),
                            prob_u_teacher.detach(),
                            low_mask_all,
                            high_mask_all,
                            cfg_contra,
                            memobank,
                            queue_ptrlis,
                            queue_size,
                            rep_all_teacher.detach(),
                        )
                    else:
                        prototype, new_keys, contra_loss = compute_contra_memobank_loss(
                            rep_all,
                            label_l_small.long(),
                            label_u_small.long(),
                            prob_l_teacher.detach(),
                            prob_u_teacher.detach(),
                            low_mask_all,
                            high_mask_all,
                            cfg_contra,
                            memobank,
                            queue_ptrlis,
                            queue_size,
                            rep_all_teacher.detach(),
                            prototype,
                        )

                dist.all_reduce(contra_loss)
                contra_loss = (
                        contra_loss
                        / world_size
                        * cfg["trainer"]["contrastive"].get("loss_weight", 1)
                )

            else:
                contra_loss = 0 * rep_all.sum()
                
            # partial part    
            with torch.no_grad():
                pred_p_w = model(image_p)["pred"]
                pred_p_w = F.interpolate(
                    pred_p_w, size=(h, w), mode="bilinear", align_corners=True
                )
                # get the pseudo label and the corresponding confidence, label_p_w is the pseudo label
                conf_p_w = pred_p_w.softmax(dim=1).max(dim=1)[0]
                label_p_w = pred_p_w.argmax(dim=1)  # (batch, 256, 256)
                
            # mask_p_label: if the pixel is labeled as 1 or 2 in the partial gt label
            mask_p_label = (label_p == 1) | (label_p == 2)
            # mask_p_unlabel: other pixels in the partial gt label
            mask_p_unlabel = ~mask_p_label
            # mask_p_unlabel_high_conf: pixels labeled as 0 and 3 in the pseudo label with high confidence
            mask_p_unlabel_high_conf = mask_p_unlabel & ((label_p_w == 0) | (label_p_w == 3)) &\
                                       (conf_p_w >= cfg["trainer"]["partial_supervised"].get("conf_threshold", 0.9))
            # mask_p_unlabel_others: pixels labeled as 0 and 3 in the pseudo label with low confidence & pixels
            # labeled as 1 and 2 in the pseudo label
            mask_p_unlabel_others = mask_p_unlabel ^ mask_p_unlabel_high_conf
 
            # criterion for supervised loss
            criterion_partial = torch.nn.CrossEntropyLoss(reduction='none')

            # supervised loss for label 1 and 2
            partial_sup_loss = criterion_partial(pred_p_s, label_p)  # [batch, 256, 256]
            partial_sup_loss = partial_sup_loss * mask_p_label
            partial_sup_loss = partial_sup_loss.sum() / (mask_p_label.sum() + 1)

            # loss for label 0 and 3 with high confidence
            partial_unsup_loss_1 = criterion_partial(pred_p_s, label_p_w)
            partial_unsup_loss_1 = partial_unsup_loss_1 * mask_p_unlabel_high_conf
            partial_unsup_loss_1 = partial_unsup_loss_1.sum() / (mask_p_unlabel_high_conf.sum() + 1)

            # exclusive loss for other pixels(we don't want these pixels predicted as label 1 and 2)

            partial_unsup_loss_2 = exclusive_loss(pred_p_s, exclude_label=[0, 1, 2])
            partial_unsup_loss_2 = partial_unsup_loss_2 * mask_p_unlabel_others
            partial_unsup_loss_2 = partial_unsup_loss_2.sum() / (mask_p_unlabel_others.sum() + 1)

            partial_loss = partial_sup_loss + partial_unsup_loss_1 + partial_unsup_loss_2   
                

        # loss = sup_loss + unsup_loss + contra_loss
        loss = sup_loss + unsup_loss + contra_loss + partial_loss
        
        # dist.barrier()

        
        confidence_threshold = cfg["trainer"]["unsupervised"].get("confidence_threshold", 0.9)
        
        # ===================== 有标签数据的精炼 =====================
        # 获取学生模型在有标签数据上的预测结果，并计算预测标签
        pred_l_labels = torch.argmax(pred_l_large.detach(), dim=1)  # (batch_size, h, w)
        
        # 将预测标签和真实标签转换为CPU上的numpy数组
        pred_l_labels_np = pred_l_labels.cpu().numpy()
        label_l_np = label_l.cpu().numpy()
        
        # 使用color_map将预测标签和真实标签映射为彩色图像
        pred_l_color = label_embed(color_map, pred_l_labels_np)  # 学生模型预测的彩色标签
        label_l_color = label_embed(color_map, label_l_np)       # 真实的彩色标签
        
        # 计算Dice系数t，衡量预测与真实标签的相似度
        t = dice_refine(
            inputs=pred_l_labels.unsqueeze(1),
            target=label_l.unsqueeze(1),
            oh_input=True,
            ignore = (logits_u_aug<confidence_threshold),
            weight=[0,0.2,0.2,0.6]
        )
        
        # t是一个标量，需要为每个样本创建一个张量，并放大系数
        t = torch.ones((pred_l_labels.shape[0]), dtype=torch.float32, device=pred_l_labels.device) * t * 999 * 4
        
        # 将预测的彩色标签、Dice系数t、原始图像和真实的彩色标签输入到精炼模型中进行训练
        lat_loss_sup, ref_outputs = refine_model(
            pred_l_color.cuda(),
            t,
            image_l.cuda(),
            training=True,
            good=label_l_color.cuda(),
        )
        
        # 计算精炼后的输出的softmax值
        ref_outputs_soft = torch.softmax(ref_outputs, dim=1)
        
        # 计算有监督精炼损失，包括交叉熵损失和Dice损失，以及潜在损失
        sup_loss_cedice = (
            0.5 * criterion_refine_ce(ref_outputs, label_l.long())
            + 0.5 * criterion_dice(ref_outputs_soft, label_l.unsqueeze(1).float())
        )
        sup_loss_ref = sup_loss_cedice + lat_loss_sup
        
        # ===================== 无标签数据的精炼 =====================
        # 获取教师模型在无标签数据上的预测结果，并计算预测标签和置信度
        # # 置信度和伪标签
        
        # 创建高置信度的掩码
        # confidence_mask = (logits_u_aug >= confidence_threshold)
        
        # # 将低置信度的像素标记为无效标签（设为255）
        # label_u_aug[~confidence_mask] = 255
        
        # # 获取学生模型在无标签数据上的预测结果，并计算预测标签
        # pred_u_labels = torch.argmax(pred_u_large.detach(), dim=1)
        
        # # 将预测标签和伪标签转换为CPU上的numpy数组
        # pred_u_labels_np = pred_u_labels.cpu().numpy()
        # pseudo_labels_u_np = label_u_aug.cpu().numpy()
        
        # # 使用color_map将预测标签和伪标签映射为彩色图像
        # pred_u_color = label_embed(color_map, pred_u_labels_np)          # 学生模型预测的彩色标签
        # pseudo_labels_u_color = label_embed(color_map, pseudo_labels_u_np)  # 教师模型伪标签的彩色图像
        
        # # 计算Dice系数t2，衡量学生模型预测与教师模型伪标签的相似度
        # t2 = dice_refine(
        #     inputs=pred_u_labels.unsqueeze(1),
        #     target=label_u_aug.unsqueeze(1),
        #     oh_input=True,
        #     ignore = (logits_u_aug<confidence_threshold),
        #     weight = [0,0.2,0.2,0.6]
        # )
        
        # # t2是一个标量，需要为每个样本创建一个张量，并放大系数
        # t2 = torch.ones((pred_u_labels.shape[0]), dtype=torch.float32, device=pred_u_labels.device) * t2 * 999 * 4
        
        # # 将预测的彩色标签、Dice系数t2、原始图像和伪标签的彩色图像输入到精炼模型中进行训练
        # lat_loss_unsup, ref_outputs_u = refine_model(
        #     pred_u_color.cuda(),
        #     t2,
        #     image_u.cuda(),
        #     training=True,
        #     good=pseudo_labels_u_color.cuda(),
        # )
        
        # # 计算精炼后的输出的softmax值
        # ref_outputs_u_soft = torch.softmax(ref_outputs_u, dim=1)
        
        # ===================== 计算无监督精炼损失（避免使用索引） =====================
        # 创建有效像素的掩码
        prob_u_teacher = F.softmax(pred_u_large_teacher.detach(), dim=1)
        confidences_u, pseudo_labels_u = torch.max(prob_u_teacher, dim=1)  # 置信度和伪标签
        
        # 获取学生模型在无标签数据上的预测结果，并计算预测标签
        pred_u_labels = torch.argmax(pred_u_large.detach(), dim=1)
        
        # 将预测标签和伪标签转换为CPU上的numpy数组
        pred_u_labels_np = pred_u_labels.cpu().numpy()
        pseudo_labels_u_np = pseudo_labels_u.cpu().numpy()
        
        # 使用color_map将预测标签和伪标签映射为彩色图像
        pred_u_color = label_embed(color_map, pred_u_labels_np)          # 学生模型预测的彩色标签
        pseudo_labels_u_color = label_embed(color_map, pseudo_labels_u_np)  # 教师模型伪标签的彩色图像
        
        # 计算Dice系数t2，衡量学生模型预测与教师模型伪标签的相似度
        t2 = dice_refine(
            inputs=pred_u_labels.unsqueeze(1),
            target=pseudo_labels_u.unsqueeze(1),
            oh_input=True,
        )
        
        # t2是一个标量，需要为每个样本创建一个张量，并放大系数
        t2 = torch.ones((pred_u_labels.shape[0]), dtype=torch.float32, device=pred_u_labels.device) * t2 * 999
        
        # 将预测的彩色标签、Dice系数t2、原始图像和伪标签的彩色图像输入到精炼模型中进行训练
        lat_loss_unsup, ref_outputs_u = refine_model(
            pred_u_color.cuda(),
            t2,
            image_u.cuda(),
            training=True,
            good=pseudo_labels_u_color.cuda(),
        )
        
        # 计算精炼后的输出的softmax值
        ref_outputs_u_soft = torch.softmax(ref_outputs_u, dim=1)
        
        # ===================== 使用pseudo_mask计算无监督精炼损失 =====================
        # 计算normalize后的置信度图
        ref_outputs_u_soft_norm = normalize(ref_outputs_u_soft)
        
        # 创建pseudo_mask，阈值为confidence_threshold
        pseudo_mask = (ref_outputs_u_soft_norm > confidence_threshold).float()
        
        # 将ref_outputs_u_soft与pseudo_mask相乘
        ref_outputs_u_soft_masked = ref_outputs_u_soft * pseudo_mask
        
        # 从mask后的输出中获取伪标签
        pseudo_outputs_ref = torch.argmax(ref_outputs_u_soft_masked.detach(), dim=1)
        
        # 计算交叉熵损失
        unsup_loss_ce = criterion_refine_ce(ref_outputs_u, pseudo_outputs_ref.long())
        # 将损失乘以pseudo_mask（需要调整维度）
        unsup_loss_ce = unsup_loss_ce * pseudo_mask.squeeze(1)
        # 计算平均损失
        unsup_loss_ce = unsup_loss_ce.sum() / (pseudo_mask.sum() + 1e-10)
        
        # 计算Dice损失
        unsup_loss_dice = criterion_dice(ref_outputs_u_soft, pseudo_outputs_ref.unsqueeze(1).float())
        unsup_loss_dice = unsup_loss_dice * pseudo_mask
        unsup_loss_dice = unsup_loss_dice.sum() / (pseudo_mask.sum() + 1e-10)
        
        # 总的无监督精炼损失
        unsup_loss_cedice = 0.5 * unsup_loss_ce + 0.5 * unsup_loss_dice
        unsup_loss_ref = unsup_loss_cedice + lat_loss_unsup
        
        label_mix = mix_label(label_p_w, pred_p_w)

            # NOTE:正常
        pred_p_s_soft = torch.softmax(pred_p_s, dim=1)
        mask_p_s = normalize(pred_p_s_soft) > confidence_threshold
        conf_pseudo_p_s = pred_p_s_soft * mask_p_s
        pseudo_p_s = torch.argmax(conf_pseudo_p_s.detach(), dim=1, keepdim=False)
        refine_p = pseudo_p_s.detach().clone()  # lab + unlab

        color_refine_p = pl_weak_embed(color_map, refine_p.cpu().numpy())
        partial_label_batch_numpy = label_mix.cpu().numpy()
        partial_label_batch_color = pl_weak_embed(color_map, partial_label_batch_numpy)
        t_partial = dice_refine(inputs=refine_p.unsqueeze(1), target=label_mix.unsqueeze(1), oh_input=True,weight=[0,0.2,0.2,0.6])
        t_partial = torch.ones((color_refine_p.shape[0]), dtype=torch.float32, device="cuda") * t_partial * 999 *4
        lat_loss_sup, ref_outputs_partial = refine_model(
            color_refine_p.cuda(), t_partial, image_p.cuda(), training=True, good=partial_label_batch_color.cuda()
        )
        ref_outputs_partial_soft = torch.softmax(ref_outputs_partial,dim=1)
        sup_loss_cedice_partial = 0.5 * F.cross_entropy(
            ref_outputs_partial,
            label_mix.long(),
            ignore_index=255,  # weight=torch.tensor([0, 1, 1, 0], dtype=torch.float, device="cuda")
        ) + 0.5 * criterion_dice(ref_outputs_partial_soft,label_mix.unsqueeze(1).float())
        partial_loss_ref = sup_loss_cedice_partial + lat_loss_sup
        
        # ===================== 更新精炼模型 =====================
        # 计算总的精炼损失
        refine_loss = sup_loss_ref + unsup_loss_ref + partial_loss_ref
        
        # 优化精炼模型的参数
        refine_optimizer.zero_grad()
        refine_loss.backward()
        refine_optimizer.step()

        # ===================== 模型校正损失（在epoch > 10时进行） =====================
        if epoch > 10:
            # 重新计算 t
            t = dice_refine(
                inputs=pred_l_labels.unsqueeze(1),
                target=label_l.unsqueeze(1),
                oh_input=True,
                weight=[0,0.2,0.2,0.6]
            )
            t = torch.ones((pred_l_labels.shape[0]), dtype=torch.float32, device=pred_l_labels.device) * t * 999 * 4

            # 精炼模型推理
            ref_outputs = refine_model(
                pred_l_color.cuda(),
                t,
                image_l.cuda(),
                training=False,
            )

            ref_outputs_soft_for_refine = torch.softmax(ref_outputs, dim=1)
            # 使用阈值过滤
            pseudo_mask = (normalize(ref_outputs_soft_for_refine) > confidence_threshold).float()
            ref_outputs_soft_masked = ref_outputs_soft_for_refine * pseudo_mask
            pseudo_outputs_ref = torch.argmax(ref_outputs_soft_masked.detach(), dim=1)

            # 计算校正损失
            pred_l_updated = model(image_l)["pred"]
            pred_l_updated = F.interpolate(pred_l_updated, size=(h, w), mode="bilinear", align_corners=True)
            # 计算损失
            loss_rect_ce = criterion_refine_ce(pred_l_updated, pseudo_outputs_ref.long())
            loss_rect_ce = loss_rect_ce * pseudo_mask.squeeze(1)
            loss_rect_ce = loss_rect_ce.sum() / (pseudo_mask.sum() + 1e-10)
            loss_rect_dice = criterion_dice(pred_l_updated.softmax(dim=1), pseudo_outputs_ref.unsqueeze(1).float())
            loss_rect_dice = loss_rect_dice * pseudo_mask
            loss_rect_dice = loss_rect_dice.sum() / (pseudo_mask.sum() + 1e-10)
            loss_rect = 0.5 * loss_rect_ce + 0.5 * loss_rect_dice

            # 优化分割模型
            
            total_loss = loss + loss_rect
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
        else :
            total_loss = loss
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

        # update teacher model with EMA
        if epoch >= cfg["trainer"].get("sup_only_epoch", 1):
            with torch.no_grad():
                ema_decay = min(
                    1
                    - 1
                    / (
                            i_iter
                            - len(loader_l) * cfg["trainer"].get("sup_only_epoch", 1)
                            + 1
                    ),
                    ema_decay_origin,
                )
                for t_params, s_params in zip(
                        model_teacher.parameters(), model.parameters()
                ):
                    t_params.data = (
                            ema_decay * t_params.data + (1 - ema_decay) * s_params.data
                    )

        # gather all loss from different gpus
        reduced_sup_loss = sup_loss.clone().detach()
        dist.all_reduce(reduced_sup_loss)
        sup_losses.update(reduced_sup_loss.item())

        reduced_uns_loss = unsup_loss.clone().detach()
        dist.all_reduce(reduced_uns_loss)
        uns_losses.update(reduced_uns_loss.item())

        reduced_con_loss = contra_loss.clone().detach()
        dist.all_reduce(reduced_con_loss)
        con_losses.update(reduced_con_loss.item())

        reduced_partial_loss = partial_loss.clone().detach()
        dist.all_reduce(reduced_partial_loss)
        partial_losses.update(reduced_partial_loss.item())
        
        reduced_partial_sup_loss = partial_sup_loss.clone().detach()
        dist.all_reduce(reduced_partial_sup_loss)
        partial_sup_losses.update(reduced_partial_sup_loss.item())
        
        reduced_partial_unsup_loss_1 = partial_unsup_loss_1.clone().detach()
        dist.all_reduce(reduced_partial_unsup_loss_1)
        partial_unsup_losses_1.update(reduced_partial_unsup_loss_1.item())
        
        reduced_partial_unsup_loss_2 = partial_unsup_loss_2.clone().detach()
        dist.all_reduce(reduced_partial_unsup_loss_2)
        partial_unsup_losses_2.update(reduced_partial_unsup_loss_2.item())

        batch_end = time.time()
        batch_times.update(batch_end - batch_start)

        if i_iter % 10 == 0 and rank == 0:
            logger.info(
                "[{}] "
                "Iter [{}/{}]\t"
                "Data {data_time.val:.2f} ({data_time.avg:.2f})\t"
                "Time {batch_time.val:.2f} ({batch_time.avg:.2f})\t"
                "Sup {sup_loss.val:.3f} ({sup_loss.avg:.3f})\t"
                "Uns {uns_loss.val:.3f} ({uns_loss.avg:.3f})\t"
                "Con {con_loss.val:.3f} ({con_loss.avg:.3f})\t"
                "Partial {partial_loss.val:.3f} ({partial_loss.avg:.3f})\t"
                "Partial_Sup {partial_sup_loss.val:.3f} ({partial_sup_loss.avg:.3f})\t"
                "Partial_Unsup_1 {partial_unsup_loss_1.val:.3f} ({partial_unsup_loss_1.avg:.3f})\t"
                "Partial_Unsup_2 {partial_unsup_loss_2.val:.3f} ({partial_unsup_loss_2.avg:.3f})\t"
                "LR {lr.val:.5f}".format(
                    contra_flag,
                    i_iter,
                    cfg["trainer"]["epochs"] * len(loader_l),
                    data_time=data_times,
                    batch_time=batch_times,
                    sup_loss=sup_losses,
                    uns_loss=uns_losses,
                    con_loss=con_losses,
                    partial_loss=partial_losses,
                    partial_sup_loss=partial_sup_losses,
                    partial_unsup_loss_1=partial_unsup_losses_1,
                    partial_unsup_loss_2=partial_unsup_losses_2,
                    lr=learning_rates,
                )
            )

            tb_logger.add_scalar("lr", learning_rates.val, i_iter)
            tb_logger.add_scalar("Sup Loss", sup_losses.val, i_iter)
            tb_logger.add_scalar("Uns Loss", uns_losses.val, i_iter)
            tb_logger.add_scalar("Con Loss", con_losses.val, i_iter)
            tb_logger.add_scalar("Partial Loss", partial_losses.val, i_iter)
            tb_logger.add_scalar("Partial Sup Loss", partial_sup_losses.val, i_iter)
            tb_logger.add_scalar("Partial Unsup Loss 1", partial_unsup_losses_1.val, i_iter)
            tb_logger.add_scalar("Partial Unsup Loss 2", partial_unsup_losses_2.val, i_iter)
        # dist.barrier()


def validate(
        model,
        data_loader,
        epoch,
        logger,
):
    model.eval()
    data_loader.sampler.set_epoch(epoch)

    num_classes, ignore_label = (
        cfg["net"]["num_classes"],
        cfg["dataset"]["ignore_label"],
    )
    rank, world_size = dist.get_rank(), dist.get_world_size()

    intersection_meter = AverageMeter()
    union_meter = AverageMeter()

    for step, batch in enumerate(data_loader):
        images, labels = batch
        images = images.cuda()
        labels = labels.long().cuda()

        with torch.no_grad():
            outs = model(images)

        # get the output produced by model_teacher
        output = outs["pred"]
        output = F.interpolate(
            output, labels.shape[1:], mode="bilinear", align_corners=True
        )
        output = output.data.max(1)[1].cpu().numpy()
        target_origin = labels.cpu().numpy()

        # start to calculate miou
        intersection, union, target = intersectionAndUnion(
            output, target_origin, num_classes, ignore_label
        )

        # gather all validation information
        reduced_intersection = torch.from_numpy(intersection).cuda()
        reduced_union = torch.from_numpy(union).cuda()
        reduced_target = torch.from_numpy(target).cuda()

        dist.all_reduce(reduced_intersection)
        dist.all_reduce(reduced_union)
        dist.all_reduce(reduced_target)

        intersection_meter.update(reduced_intersection.cpu().numpy())
        union_meter.update(reduced_union.cpu().numpy())

    iou_class = intersection_meter.sum / (union_meter.sum + 1e-10)
    mIoU = np.mean(iou_class)
    mIoU_without_bg = np.mean(iou_class[1:])

    if rank == 0:
        for i, iou in enumerate(iou_class):
            logger.info(" * class [{}] IoU {:.2f}".format(i, iou * 100))
        logger.info(" * epoch {} mIoU {:.2f}".format(epoch, mIoU * 100))
        logger.info(" * epoch {} mIoU_without_bg {:.2f}".format(epoch, mIoU_without_bg * 100))
    # dist.barrier()

    return mIoU_without_bg, iou_class[3]

def normalize(tensor):
        min_val = tensor.min(1, keepdim=True)[0]
        max_val = tensor.max(1, keepdim=True)[0]
        result = tensor - min_val
        result = result / max_val
        return result
def pl_weak_embed(color_map, pseudo_outputs_numpy):
    pseudo_outputs_color = torch.zeros((pseudo_outputs_numpy.shape[0], 3, pseudo_outputs_numpy.shape[1], pseudo_outputs_numpy.shape[2]), dtype=torch.float32)
    for i in range(pseudo_outputs_numpy.shape[0]):
        # Map each class value to a color value using the color map
        color_data = np.zeros((pseudo_outputs_numpy.shape[1], pseudo_outputs_numpy.shape[2], 3), dtype=np.uint8)
        for class_id, color in color_map.items():
            color_data[pseudo_outputs_numpy[i] == class_id] = color # color_data is a 2D array of RGB values, shape: (height, width, 3)
        color_image = Image.fromarray(color_data, mode="RGB")
        color_tensor = transforms.ToTensor()(color_image)
        pseudo_outputs_color[i] = color_tensor
    return pseudo_outputs_color

def pl_strong_embed(color_map, pseudo_outputs_strong_numpy):
    pseudo_outputs_strong_color = torch.zeros((pseudo_outputs_strong_numpy.shape[0], 3, pseudo_outputs_strong_numpy.shape[1], pseudo_outputs_strong_numpy.shape[2]), dtype=torch.float32)
    for i in range(pseudo_outputs_strong_numpy.shape[0]):
        color_data = np.zeros((pseudo_outputs_strong_numpy.shape[1], pseudo_outputs_strong_numpy.shape[2], 3), dtype=np.uint8)
        for class_id, color in color_map.items():
            color_data[pseudo_outputs_strong_numpy[i] == class_id] = color
        color_image = Image.fromarray(color_data, mode="RGB")
        color_tensor = transforms.ToTensor()(color_image)
        pseudo_outputs_strong_color[i] = color_tensor
    return pseudo_outputs_strong_color
def mix_label(label, pred, conf_threshold: float = 0.95):
    pred_soft = torch.softmax(pred, dim=1)
    mask_pred = normalize(pred_soft) > conf_threshold
    conf_pseudo_p_w = pred_soft * mask_pred
    pseudo_label = torch.argmax(conf_pseudo_p_w.detach(), dim=1, keepdim=False)
    mask = (label == 1) | (label == 2)
    pseudo_mask = (pseudo_label == 1) | (pseudo_label == 2)
    pseudo_label[pseudo_mask] = 0
    pseudo_label[mask] = label[mask]
    return pseudo_label.detach()

def label_embed(color_map, label_batch_numpy):
    label_batch_color = torch.zeros((label_batch_numpy.shape[0], 3, label_batch_numpy.shape[1], label_batch_numpy.shape[2]), dtype=torch.float32, device='cuda')
    for i in range(label_batch_numpy.shape[0]):
        color_data = np.zeros((label_batch_numpy.shape[1], label_batch_numpy.shape[2], 3), dtype=np.uint8)
        for class_id, color in color_map.items():
            color_data[label_batch_numpy[i] == class_id] = color
        color_image = Image.fromarray(color_data, mode="RGB")
        color_tensor = transforms.ToTensor()(color_image)
        label_batch_color[i] = color_tensor
    return label_batch_color

if __name__ == "__main__":
    main()
