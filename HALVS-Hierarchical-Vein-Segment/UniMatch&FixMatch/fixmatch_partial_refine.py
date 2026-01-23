import argparse
import logging
import os
import pprint
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from torch.optim import SGD
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import yaml
from torch import nn, Tensor
from dataset.semi import SemiDataset
from model.semseg.deeplabv3plus import DeepLabV3Plus
from supervised import evaluate
from util.classes import CLASSES
from util.ohem import ProbOhemCrossEntropy2d
from util.utils import count_params, init_log, AverageMeter
from util.dist_helper import setup_distributed, exclusive_loss
from util.loss import DiceLoss, DiceLoss_refine
from torchvision import transforms
from PIL import Image
from model.unet_de import UNet_LDMV2
from torch.nn import functional as F
from torch.utils.data.distributed import DistributedSampler

cudnn.enabled = True
cudnn.benchmark = True
color_map = {
    0: (0, 0, 0),  # background class
    1: (255, 0, 0),  # class 1 (red)
    2: (0, 255, 0),  # class 2 (green)
    3: (0, 0, 255),  # class 3 (blue)
}


def get_args():
    parser = argparse.ArgumentParser(description="Revisiting Weak-to-Strong Consistency in Semi-Supervised Semantic Segmentation")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--labeled-id-path", type=str, required=True)
    parser.add_argument("--partial_labeled_id_path", type=str, required=True)
    parser.add_argument("--unlabeled-id-path", type=str, required=True)
    parser.add_argument("--save-path", type=str, required=True)
    parser.add_argument("--local_rank", default=0, type=int)
    parser.add_argument("--port", default=None, type=int)
    return parser.parse_args()


def main():
    args = get_args()
    cfg = yaml.load(open(args.config, "r"), Loader=yaml.Loader)
    logger = init_log("global", logging.INFO)
    rank, world_size = setup_distributed(port=args.port)
    local_rank = int(os.environ["LOCAL_RANK"])
    if rank == 0:
        all_args = {**cfg, **vars(args), "ngpus": world_size}
        logger.info("{}\n".format(pprint.pformat(all_args)))
        writer = SummaryWriter(args.save_path)
        os.makedirs(args.save_path, exist_ok=True)

    model = DeepLabV3Plus(cfg)
    if rank == 0:
        logger.info("Total params: {:.1f}M\n".format(count_params(model)))
    optimizer = SGD(
        [
            {"params": model.backbone.parameters(), "lr": cfg["lr"]},
            {"params": [param for name, param in model.named_parameters() if "backbone" not in name], "lr": cfg["lr"] * cfg["lr_multi"]},
        ],
        lr=cfg["lr"],
        momentum=0.9,
        weight_decay=1e-4,
    )
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model.cuda()

    refine_model = UNet_LDMV2(in_chns=3 + 3, class_num=4, out_chns=4, ldm_method="replace", ldm_beta_sch="cosine", ts=10, ts_sample=2).cuda()
    model = nn.parallel.DistributedDataParallel(
        model, device_ids=[local_rank], broadcast_buffers=False, output_device=local_rank, find_unused_parameters=True
    )
    refine_optimizer = SGD(refine_model.parameters(), lr=cfg["lr"], momentum=0.9, weight_decay=0.0001)

    # NOTE: criterion
    if cfg["criterion"]["name"] == "CELoss":
        criterion_l = nn.CrossEntropyLoss(**cfg["criterion"]["kwargs"]).cuda(local_rank)
    elif cfg["criterion"]["name"] == "OHEM":
        criterion_l = ProbOhemCrossEntropy2d(**cfg["criterion"]["kwargs"]).cuda(local_rank)
    else:
        raise NotImplementedError("%s criterion is not implemented" % cfg["criterion"]["name"])
    criterion_u = nn.CrossEntropyLoss(reduction="none").cuda(local_rank)
    criterion_dice = DiceLoss(n_classes=4)
    dice_refine = DiceLoss_refine(n_classes=4)
    criterion_refine_ce = nn.CrossEntropyLoss()

    # NOTE: dataset
    trainset_u = SemiDataset(cfg["dataset"], cfg["data_root"], "train_u", cfg["crop_size"], args.unlabeled_id_path)
    trainset_l = SemiDataset(cfg["dataset"], cfg["data_root"], "train_l", cfg["crop_size"], args.labeled_id_path, nsample=len(trainset_u.ids))
    trainset_p = SemiDataset(cfg["dataset"], cfg["data_root"], "train_p", cfg["crop_size"], args.partial_labeled_id_path, nsample=len(trainset_u.ids))
    valset = SemiDataset(cfg["dataset"], cfg["data_root"], "val")
    trainsampler_l = DistributedSampler(trainset_l)
    trainloader_l = DataLoader(trainset_l, batch_size=cfg["batch_size"], pin_memory=True, num_workers=1, drop_last=True, sampler=trainsampler_l)
    trainsampler_u = DistributedSampler(trainset_u)
    trainloader_u = DataLoader(trainset_u, batch_size=cfg["batch_size"], pin_memory=True, num_workers=1, drop_last=True, sampler=trainsampler_u)
    trainsampler_p = DistributedSampler(trainset_p)
    trainloader_p = DataLoader(trainset_p, batch_size=cfg["batch_size"], pin_memory=True, num_workers=1, drop_last=True, sampler=trainsampler_p)
    valsampler = DistributedSampler(valset)
    valloader = DataLoader(valset, batch_size=1, pin_memory=True, num_workers=1, drop_last=False, sampler=valsampler)

    total_iters = len(trainloader_u) * cfg["epochs"]
    previous_best = 0.0
    past_max_iou = 0.0
    epoch = -1

    if os.path.exists(os.path.join(args.save_path, "latest.pth")):
        checkpoint = torch.load(os.path.join(args.save_path, "latest.pth"))
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        epoch = checkpoint["epoch"]
        previous_best = checkpoint["previous_best"]
        if rank == 0:
            logger.info(f"Load from checkpoint at epoch {epoch}\n")

    for epoch in range(epoch + 1, cfg["epochs"]):
        if rank == 0:
            logger.info("===========> Epoch: {:}, LR: {:.5f}, Previous best: {:.2f}".format(epoch, optimizer.param_groups[0]["lr"], previous_best))
        total_loss = AverageMeter()
        total_loss_u_p = AverageMeter()
        total_loss_x = AverageMeter()
        total_loss_s = AverageMeter()
        total_loss_p = AverageMeter()
        total_loss_p_sup_loss = AverageMeter()
        total_loss_p_unsup_loss_1 = AverageMeter()
        total_loss_p_unsup_loss_2 = AverageMeter()
        total_mask_ratio = AverageMeter()
        trainloader_l.sampler.set_epoch(epoch)
        trainloader_u.sampler.set_epoch(epoch)
        trainloader_p.sampler.set_epoch(epoch)
        loader: list[list[Tensor]] = zip(trainloader_l, trainloader_p, trainloader_u, trainloader_u)

        for i, (
            (img_x, mask_x),
            (img_p_w, mask_p, img_p_s),
            (img_u_w, img_u_s, _, ignore_mask, cutmix_box, _),
            (img_u_w_mix, img_u_s_mix, _, ignore_mask_mix, _, _),
        ) in enumerate(loader):

            img_x, mask_x = img_x.cuda(), mask_x.cuda()
            img_p_w, mask_p, img_p_s = img_p_w.cuda(), mask_p.cuda(), img_p_s.cuda()
            img_u_w, img_u_s = img_u_w.cuda(), img_u_s.cuda()
            ignore_mask, cutmix_box = ignore_mask.cuda(), cutmix_box.cuda()
            img_u_w_mix, img_u_s_mix = img_u_w_mix.cuda(), img_u_s_mix.cuda()
            ignore_mask_mix = ignore_mask_mix.cuda()
            num_lb, num_ulb = img_x.shape[0], img_u_w.shape[0]  # label_batch和unlabel_batch

            with torch.no_grad():
                model.eval()
                pred_u_w_mix = model(img_u_w_mix).detach()
                conf_u_w_mix = pred_u_w_mix.softmax(dim=1).max(dim=1)[0]  # 概率值
                mask_u_w_mix = pred_u_w_mix.argmax(dim=1)  # 伪标签
                pred_p_w = model(img_p_w)
                # print(pred_p_w.shape)
                conf_p_w = pred_p_w.softmax(dim=1).max(dim=1)[0]  # (B,H,W)
                label_p_w = pred_p_w.argmax(dim=1)  # 弱增强后的当作伪标签

            img_u_s[cutmix_box.unsqueeze(1).expand(img_u_s.shape) == 1] = img_u_s_mix[
                cutmix_box.unsqueeze(1).expand(img_u_s.shape) == 1
            ]  # 将 img_u_s_mix 图像中对应 cutmix_box == 1 的区域像素值，替换到 img_u_s 的相同位置。

            model.train()
            torch.set_grad_enabled(True)

            pred_p_s = model(img_p_s)  # partial图片强增强预测
            pred_x, pred_u_w = model(torch.cat((img_x, img_u_w))).split([num_lb, num_ulb])  # 生成有监督和无监督弱增强的对应预测
            pred_x_soft = torch.softmax(pred_x, dim=1)
            pseudo_mask = normalize(pred_x_soft) > cfg["conf_threshold"]
            pred_mask = pred_x_soft * pseudo_mask
            pred_x_output = torch.argmax(pred_mask.detach(), dim=1, keepdim=False)

            pred_u_s = model(img_u_s)  # 生成强增强的对应预测
            pred_u_w = pred_u_w.detach()
            conf_u_w = pred_u_w.softmax(dim=1).max(dim=1)[0]
            # print(conf_u_w.shape)
            mask_u_w = pred_u_w.argmax(dim=1)  # 弱增强后当作伪标签
            mask_u_w_cutmixed, conf_u_w_cutmixed, ignore_mask_cutmixed = mask_u_w.clone(), conf_u_w.clone(), ignore_mask.clone()

            mask_u_w_cutmixed[cutmix_box == 1] = mask_u_w_mix[cutmix_box == 1]
            conf_u_w_cutmixed[cutmix_box == 1] = conf_u_w_mix[cutmix_box == 1]
            ignore_mask_cutmixed[cutmix_box == 1] = ignore_mask_mix[cutmix_box == 1]

            mask_u_w_cutmixed_forref = mask_u_w_cutmixed * (conf_u_w_cutmixed >= cfg["conf_threshold"])

            pred_u_s_soft = pred_u_s.softmax(dim=1)
            pseudo_strong = normalize(pred_u_s_soft) > cfg["conf_threshold"]
            outputs_strong_masked = pred_u_s_soft * pseudo_strong
            pseudo_outputs_strong = torch.argmax(outputs_strong_masked.detach(), dim=1, keepdim=False)

            # print(mask_x.shape)
            loss_x = criterion_l(pred_x, mask_x)  # criterion_dice(pred_x.softmax(dim=1), mask_x.unsqueeze(1).float()))/2.0

            loss_u_s = criterion_u(
                pred_u_s, mask_u_w_cutmixed
            )  # + criterion_dice(pred_u_s.softmax(dim=1), mask_u_w_cutmixed.unsqueeze(1).float()))/2.0
            loss_u_s = loss_u_s * ((conf_u_w_cutmixed >= cfg["conf_thresh"]) & (ignore_mask_cutmixed != 255))
            loss_u_s = loss_u_s.sum() / (ignore_mask_cutmixed != 255).sum().item()

            loss_u_p = loss_x + loss_u_s

            mask_p_label = (mask_p == 1) | (mask_p == 2)  # bool变量矩阵
            mask_p_unlabel = ~mask_p_label  # 标签中0的索引
            mask_p_unlabel_high_conf = (
                mask_p_unlabel & ((label_p_w == 0) | (label_p_w == 3)) & (conf_p_w >= cfg["conf_threshold"])
            )  # 预测成0，3类别且高置信度>0.85
            mask_p_unlabel_others = mask_p_unlabel ^ mask_p_unlabel_high_conf
            # criterion for supervised loss
            criterion_partial = torch.nn.CrossEntropyLoss(reduction="none", ignore_index=255).cuda(local_rank)

            # supervised loss for label 1 and 2

            partial_sup_loss = criterion_partial(
                pred_p_s, mask_p
            )  # +criterion_dice(pred_p_s.softmax(dim=1), mask_p.unsqueeze(1).float()))/2.0 #+ dice_loss(pred_p_s, mask_p,num_classes=4)
            partial_sup_loss = partial_sup_loss * mask_p_label
            partial_sup_loss = partial_sup_loss.sum() / (mask_p_label.sum() + 1)

            # loss for label 0 and 3 with high confidence

            partial_unsup_loss_1 = criterion_partial(
                pred_p_s, label_p_w
            )  # +criterion_dice(pred_p_s.softmax(dim=1), label_p_w.unsqueeze(1).float()))/2.0
            partial_unsup_loss_1 = partial_unsup_loss_1 * mask_p_unlabel_high_conf
            partial_unsup_loss_1 = partial_unsup_loss_1.sum() / (mask_p_unlabel_high_conf.sum() + 1)

            # exclusive loss for other pixels(we don't want these pixels predicted as label 1 and 2)

            partial_unsup_loss_2 = exclusive_loss(pred_p_s, exclude_label=[0, 1, 2])
            partial_unsup_loss_2 = partial_unsup_loss_2 * mask_p_unlabel_others
            partial_unsup_loss_2 = partial_unsup_loss_2.sum() / (mask_p_unlabel_others.sum() + 1)

            loss_p = partial_sup_loss + partial_unsup_loss_1 + partial_unsup_loss_2

            loss = loss_u_p + cfg["lambda"] * loss_p

            torch.distributed.barrier()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            pseudo_outputs_for_refine = pred_x_output.detach().clone()  # lab+unlab
            pseudo_outputs_numpy = pseudo_outputs_for_refine.clone().detach().cpu().numpy()
            pseudo_outputs_color = pl_weak_embed(color_map, pseudo_outputs_numpy)

            pseudo_outputs_for_refine_mix = mask_u_w_cutmixed_forref.detach().clone()
            pseudo_outputs_numpy_mix = pseudo_outputs_for_refine_mix.clone().detach().cpu().numpy()
            pseudo_outputs_for_refine_mix_color = pl_weak_embed(color_map, pseudo_outputs_numpy_mix)
            
            pseudo_outputs_strong_forrefine = pseudo_outputs_strong.detach().clone()
            pseudo_outputs_strong_numpy = pseudo_outputs_strong_forrefine.cpu().numpy()
            pseudo_outputs_strong_color = pl_strong_embed(color_map, pseudo_outputs_strong_numpy)

            # 2. 潜在上下文精炼模块 - 有标签数据
            label_batch_numpy = mask_x.cpu().numpy()
            label_batch_color = label_embed(color_map, label_batch_numpy)
            label_batch_color = torch.cat((label_batch_color.cuda(), pseudo_outputs_color[num_lb:].cuda()), dim=0)  # lab+unlab
            # 计算 Dice 损失
            # print(mask_x.size())
            t = dice_refine(
                inputs=pseudo_outputs_for_refine.unsqueeze(1),
                target=mask_x.unsqueeze(1),
                oh_input=True,
                weight=[0,0.2,0.2,0.6]
            )

            t = torch.ones((pseudo_outputs_color.shape[0]), dtype=torch.float32, device="cuda") * t * 999 *4

            # 精炼有标签数据
            lat_loss_sup, ref_outputs = refine_model(
                pseudo_outputs_color.cuda(),
                t,
                img_x.cuda(),
                training=True,
                good=label_batch_color.cuda(),
            )
            ref_outputs_soft = torch.softmax(ref_outputs, dim=1)

            # 计算有监督精炼损失
            sup_loss_cedice = 0.5 * criterion_l(ref_outputs[:num_lb], mask_x.long()) + 0.5 * criterion_dice(
                ref_outputs_soft[:num_lb], mask_x.unsqueeze(1).float()
            )
            sup_loss_ref = sup_loss_cedice + lat_loss_sup
            # 使用精炼后的输出生成新的伪标签
            ref_soft = ref_outputs_soft
            # ref_pseudo_mask = (normalize(ref_soft) > cfg['conf_thresh']).float()
            # ref_outputs_masked = ref_soft * ref_pseudo_mask
            ref_pseudo_outputs = pseudo_outputs_for_refine_mix  # lab+unlab

            # 潜在上下文精炼模块 - 无标签数据
            t2 = dice_refine(pseudo_outputs_strong_forrefine.unsqueeze(1), ref_pseudo_outputs.unsqueeze(1), oh_input=True,weight=[0,0.2,0.2,0.6])

            t2 = torch.ones((pseudo_outputs_strong_color.shape[0]), dtype=torch.float32, device="cuda") * t2 * 999 * 4

            lat_loss_unsup, ref_outputs_strong = refine_model(
                pseudo_outputs_strong_color.cuda(),
                t2,
                img_u_s.cuda(),
                training=True,
                good=pseudo_outputs_for_refine_mix_color.cuda(),
            )

            ref_outputs_strong_soft = torch.softmax(ref_outputs_strong, dim=1)

            # 计算复合损失
            # ref_comp_loss, ref_as_weight = get_comp_loss(weak=ref_soft, strong=ref_outputs_strong_soft)
            unsup_loss_cedice = (
                0.5 * criterion_refine_ce(ref_outputs_strong, ref_pseudo_outputs)
                + 0.5 * criterion_dice(ref_outputs_strong_soft, ref_pseudo_outputs.unsqueeze(1))
                # + ref_as_weight * ref_comp_loss
            )

            unsup_loss_ref = unsup_loss_cedice + lat_loss_unsup
            # 计算精炼损失
            # ref_consistency_weight = (
            #     cfg['ref_consistency_weight']
            #     if cfg['ref_consistency_weight'] != -1
            #     else get_current_consistency_weight(iter_num // 150)
            # )
            # refine_loss = sup_loss_ref + ref_consistency_weight * unsup_loss_ref

            # NOTE: refine for partial
            # 1\2
            # pseudo_outputs_for_refine_partial = torch.detach(pred_p_w).clone()
            # pseudo_outputs_numpy_partial = pseudo_outputs_for_refine_partial.argmax(dim=1).cpu().numpy()
            # pseudo_outputs_color_partial = pl_weak_embed(color_map, pseudo_outputs_numpy_partial)
            # partial_label_batch_numpy = mask_p.cpu().numpy()
            # partial_label_batch_color = label_embed(color_map, partial_label_batch_numpy)
            # t_partial_1 = dice_refine(inputs=pseudo_outputs_for_refine_partial.argmax(1, True), target=mask_p.unsqueeze(1), oh_input=True)
            # t_partial_1 = torch.ones((pseudo_outputs_color_partial.shape[0]), dtype=torch.float32, device="cuda") * t_partial * 999
            # lat_loss_sup, ref_outputs = refine_model(
            #     pseudo_outputs_color_partial.cuda(), t, img_x.cuda(), training=True, good=partial_label_batch_color.cuda()
            # )
            # ref_outputs_soft = torch.softmax(ref_outputs, dim=1)
            # # 计算有监督精炼损失 TODO: only 1/2
            # sup_loss_cedice_partial = F.cross_entropy(
            #     ref_outputs, mask_p.long(), torch.tensor([0, 1, 1, 0], device="cuda"), ignore_index=255
            # )  # + 0.5 * criterion_dice(
            # # ref_outputs_soft, mask_x.unsqueeze(1).float()
            # # )
            # partail_sup_loss_ref = sup_loss_cedice_partial + lat_loss_sup

            # # 3
            # pseudo_outputs_for_refine_partial = torch.detach(pred_p_s).clone()
            # pseudo_outputs_numpy_partial = pseudo_outputs_for_refine_partial.argmax(dim=1).cpu().numpy()
            # pseudo_outputs_color_partial = pl_weak_embed(color_map, pseudo_outputs_numpy_partial)

            # ref_outputs_soft = torch.softmax(pred_p_w, dim=1)
            # pseudo_mask_partial = (normalize(ref_outputs_soft) > float(cfg["conf_thresh"])).float()
            # ref_outputs_soft_masked_partial = ref_outputs_soft * pseudo_mask_partial
            # pseudo_outputs_ref_partial = ref_outputs_soft_masked_partial.argmax(dim=1, keepdim=False)
            # partial_label_batch_numpy = pseudo_outputs_ref_partial.cpu().numpy()
            # partial_label_batch_color = label_embed(color_map, partial_label_batch_numpy)
        
            # t_partial = dice_refine(inputs=pseudo_outputs_for_refine_partial.argmax(1, True), target=pseudo_outputs_ref_partial.unsqueeze(1), oh_input=True)
            # t_partial = torch.ones((pseudo_outputs_color_partial.shape[0]), dtype=torch.float32, device="cuda") * t_partial * 999
            # lat_loss_unsup, ref_outputs = refine_model(
            #     pseudo_outputs_color_partial.cuda(), t, img_x.cuda(), training=True, good=partial_label_batch_color.cuda()
            # )
            # ref_outputs_soft = torch.softmax(ref_outputs, dim=1)
            # # 计算有监督精炼损失 TODO: only 1/2
            # unsup_loss_cedice_partial = F.cross_entropy(
            #     ref_outputs, mask_p.long(), torch.tensor([0, 1, 1, 0], device="cuda"), ignore_index=255
            # )  # + 0.5 * criterion_dice(
            # # ref_outputs_soft, mask_x.unsqueeze(1).float()
            # # )
            
            # partail_unsup_loss_ref = unsup_loss_cedice_partial + lat_loss_unsup
            label_mix = mix_label(label_p_w, pred_p_w)

            # NOTE:正常
            pred_p_s_soft = torch.softmax(pred_p_s, dim=1)
            mask_p_s = normalize(pred_p_s_soft) > cfg["conf_threshold"]
            conf_pseudo_p_s = pred_p_s_soft * mask_p_s
            pseudo_p_s = torch.argmax(conf_pseudo_p_s.detach(), dim=1, keepdim=False)
            refine_p = pseudo_p_s.detach().clone()  # lab + unlab

            color_refine_p = pl_weak_embed(color_map, refine_p.cpu().numpy())
            partial_label_batch_numpy = label_mix.cpu().numpy()
            partial_label_batch_color = pl_weak_embed(color_map, partial_label_batch_numpy)
            t_partial = dice_refine(inputs=refine_p.unsqueeze(1), target=label_mix.unsqueeze(1), oh_input=True,weight=[0,0.2,0.2,0.6])
            t_partial = torch.ones((color_refine_p.shape[0]), dtype=torch.float32, device="cuda") * t_partial * 999 * 4
            lat_loss_sup, ref_outputs_partial = refine_model(
                color_refine_p.cuda(), t_partial, img_p_w.cuda(), training=True, good=partial_label_batch_color.cuda()
            )
            ref_outputs_partial_soft = torch.softmax(ref_outputs_partial,dim=1)
            sup_loss_cedice_partial = 0.5 * F.cross_entropy(
                ref_outputs_partial,
                label_mix.long(),
                ignore_index=255,  # weight=torch.tensor([0, 1, 1, 0], dtype=torch.float, device="cuda")
            ) + 0.5 * criterion_dice(ref_outputs_partial_soft,label_mix.unsqueeze(1).float())
            partial_loss_ref = sup_loss_cedice_partial + lat_loss_sup

            refine_loss = sup_loss_ref + unsup_loss_ref + partial_loss_ref

            # 优化精炼模型
            refine_optimizer.zero_grad()
            refine_loss.backward()
            refine_optimizer.step()

            # 3. 模型校正损失
            if epoch > 10:
                # 重新计算 t
                t = dice_refine(pseudo_outputs_for_refine[:num_lb].unsqueeze(1), mask_x.unsqueeze(1), oh_input=True,weight=[0,0.2,0.2,0.6])
                t = torch.ones((pseudo_outputs_color.shape[0]), dtype=torch.float32, device="cuda") * t * 999 * 4

                # 精炼模型推理
                ref_outputs = refine_model(
                    pseudo_outputs_color.cuda(),
                    t,
                    img_x.cuda(),
                    training=False,
                )

                ref_outputs_soft_for_refine = torch.softmax(ref_outputs, dim=1)
                pseudo_mask = (normalize(ref_outputs_soft_for_refine) > cfg["conf_thresh"]).float()
                ref_outputs_soft_masked = ref_outputs_soft_for_refine * pseudo_mask
                pseudo_outputs_ref = torch.argmax(ref_outputs_soft_masked.detach(), dim=1, keepdim=False)

                # 计算校正损失
                pred_x_updated = model(img_x)
                pred_x_updated = pred_x_updated[:num_lb]
                loss_rect = 0.5 * criterion_refine_ce(pred_x_updated, pseudo_outputs_ref[:num_lb]) + 0.5 * criterion_dice(
                    pred_x_updated.softmax(dim=1), pseudo_outputs_ref[:num_lb].unsqueeze(1).float()
                )
                # 优化分割模型
                optimizer.zero_grad()
                loss_rect.backward()
                optimizer.step()

            total_loss.update(loss.item())

            total_loss_u_p.update(loss_u_p.item())
            total_loss_x.update(loss_x.item())
            total_loss_s.update(loss_u_s.item())

            total_loss_p.update(loss_p.item())
            total_loss_p_sup_loss.update(partial_sup_loss.item())
            total_loss_p_unsup_loss_1.update(partial_unsup_loss_1.item())
            total_loss_p_unsup_loss_2.update(partial_unsup_loss_2.item())

            # mask_ratio = ((conf_u_w >= cfg['conf_thresh']) & (ignore_mask != 255)).sum().item() / \
            #     (ignore_mask != 255).sum()
            # total_mask_ratio.update(mask_ratio.item())

            # iters = epoch * len(trainloader_u) + i
            # lr = cfg['lr'] * (1 - iters / total_iters) ** 0.9
            # optimizer.param_groups[0]["lr"] = lr
            # optimizer.param_groups[1]["lr"] = lr * cfg['lr_multi']

            mask_ratio = ((conf_u_w >= cfg["conf_thresh"]) & (ignore_mask != 255)).sum().item() / (ignore_mask != 255).sum()

            total_mask_ratio.update(mask_ratio.item())

            iters = epoch * len(trainloader_u) + i
            # lr = cfg['lr'] * (1 - iters / total_iters) ** 0.9
            # optimizer.param_groups[0]["lr"] = lr
            # optimizer.param_groups[1]["lr"] = lr * cfg['lr_multi']
            lr = cfg["lr"] * (1 - iters / total_iters) ** 0.9
            optimizer.param_groups[0]["lr"] = lr

            if rank == 0:
                writer.add_scalar("train/loss_all", loss.item(), iters)
                writer.add_scalar("train/loss_x", loss_x.item(), iters)
                writer.add_scalar("train/loss_s", loss_u_s.item(), iters)
                # writer.add_scalar('train/loss_w_fp', loss_u_w_fp.item(), iters)
                writer.add_scalar("train/mask_ratio", mask_ratio, iters)
                writer.add_scalar("train/loss_p", loss_p.item(), iters)
                writer.add_scalar("train/partial_sup_loss", partial_sup_loss.item(), iters)
                writer.add_scalar("train/partial_unsup_loss_1", partial_unsup_loss_1.item(), iters)
                writer.add_scalar("train/partial_unsup_loss_2", partial_unsup_loss_2.item(), iters)

            # if (i % (len(trainloader_u) // 8) == 0) and (rank == 0):
            #     logger.info('Iters: {:}, Total loss: {:.3f},Partial loss: {:.3f},Partial sup_loss: {:.3f} ,Partial unsup_loss_1: {:.3f},Partial unsup_loss_2: {:.3f},Loss x: {:.3f}, Loss s: {:.3f}, Loss w_fp: {:.3f}, Mask ratio: '
            #                 '{:.3f}'.format(i, total_loss.avg, total_loss_p.avg,total_loss_p_sup_loss.avg,total_loss_p_unsup_loss_1.avg, total_loss_p_unsup_loss_2.avg,total_loss_x.avg, total_loss_s.avg,
            #                                 total_loss_w_fp.avg, total_mask_ratio.avg))

            if (i % (len(trainloader_u) // 8) == 0) and (rank == 0):
                logger.info(
                    "Iters: {:}, Total loss: {:.3f},Partial loss: {:.3f},Partial sup_loss: {:.3f} ,Partial unsup_loss_1: {:.3f},Partial unsup_loss_2: {:.3f} ,Loss x: {:.3f}, Loss s: {:.3f}, Mask ratio: "
                    "{:.3f}".format(
                        i,
                        total_loss.avg,
                        total_loss_p.avg,
                        total_loss_p_sup_loss.avg,
                        total_loss_p_unsup_loss_1.avg,
                        total_loss_p_unsup_loss_2.avg,
                        total_loss_x.avg,
                        total_loss_s.avg,
                        total_mask_ratio.avg,
                    )
                )
        # print(loss_weights)
        eval_mode = "sliding_window" if cfg["dataset"] == "cityscapes" else "original"
        mIoU, mIoU_without_bg, iou_class = evaluate(model, valloader, eval_mode, cfg)

        iou_class_4 = iou_class[3]
        max_iou = max(iou_class_4, past_max_iou)
        past_max_iou = max_iou
        if rank == 0:
            for cls_idx, iou in enumerate(iou_class):
                logger.info("***** Evaluation ***** >>>> Class [{:} {:}] " "IoU: {:.2f}".format(cls_idx, CLASSES[cfg["dataset"]][cls_idx], iou))
            logger.info("***** Evaluation {} ***** >>>> MeanIoU: {:.2f}".format(eval_mode, mIoU))
            logger.info("***** Evaluation {} ***** >>>> MeanIoU without bg: {:.2f}".format(eval_mode, mIoU_without_bg))
            logger.info("***** Evaluation {} ***** >>>> best_3rd_IoU: {:.2f}\n".format(eval_mode, past_max_iou))

            writer.add_scalar("eval/mIoU", mIoU, epoch)
            for i, iou in enumerate(iou_class):
                writer.add_scalar("eval/%s_IoU" % (CLASSES[cfg["dataset"]][i]), iou, epoch)

        is_best = mIoU > previous_best
        previous_best = max(mIoU_without_bg, previous_best)
        if rank == 0:
            checkpoint = {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch,
                "previous_best": previous_best,
            }
            torch.save(checkpoint, os.path.join(args.save_path, "latest.pth"))
            if is_best:
                torch.save(checkpoint, os.path.join(args.save_path, "best.pth"))


def normalize(tensor):
    min_val = tensor.min(1, keepdim=True)[0]
    max_val = tensor.max(1, keepdim=True)[0]
    result = tensor - min_val
    result = result / max_val
    return result


def pl_weak_embed(color_map, pseudo_outputs_numpy):
    pseudo_outputs_color = torch.zeros(
        (pseudo_outputs_numpy.shape[0], 3, pseudo_outputs_numpy.shape[1], pseudo_outputs_numpy.shape[2]), dtype=torch.float32
    )
    for i in range(pseudo_outputs_numpy.shape[0]):
        # Map each class value to a color value using the color map
        color_data = np.zeros((pseudo_outputs_numpy.shape[1], pseudo_outputs_numpy.shape[2], 3), dtype=np.uint8)
        for class_id, color in color_map.items():
            color_data[pseudo_outputs_numpy[i] == class_id] = color  # color_data is a 2D array of RGB values, shape: (height, width, 3)
        color_image = Image.fromarray(color_data, mode="RGB")
        color_tensor = transforms.ToTensor()(color_image)
        pseudo_outputs_color[i] = color_tensor
    return pseudo_outputs_color


def pl_strong_embed(color_map, pseudo_outputs_strong_numpy):
    pseudo_outputs_strong_color = torch.zeros(
        (pseudo_outputs_strong_numpy.shape[0], 3, pseudo_outputs_strong_numpy.shape[1], pseudo_outputs_strong_numpy.shape[2]), dtype=torch.float32
    )
    for i in range(pseudo_outputs_strong_numpy.shape[0]):
        color_data = np.zeros((pseudo_outputs_strong_numpy.shape[1], pseudo_outputs_strong_numpy.shape[2], 3), dtype=np.uint8)
        for class_id, color in color_map.items():
            color_data[pseudo_outputs_strong_numpy[i] == class_id] = color
        color_image = Image.fromarray(color_data, mode="RGB")
        color_tensor = transforms.ToTensor()(color_image)
        pseudo_outputs_strong_color[i] = color_tensor
    return pseudo_outputs_strong_color


def label_embed(color_map, label_batch_numpy):
    label_batch_color = torch.zeros(
        (label_batch_numpy.shape[0], 3, label_batch_numpy.shape[1], label_batch_numpy.shape[2]), dtype=torch.float32, device="cuda"
    )
    for i in range(label_batch_numpy.shape[0]):
        color_data = np.zeros((label_batch_numpy.shape[1], label_batch_numpy.shape[2], 3), dtype=np.uint8)
        for class_id, color in color_map.items():
            color_data[label_batch_numpy[i] == class_id] = color
        color_image = Image.fromarray(color_data, mode="RGB")
        color_tensor = transforms.ToTensor()(color_image)
        label_batch_color[i] = color_tensor
    return label_batch_color
def mix_label(label: Tensor, pred: Tensor, conf_threshold: float = 0.95):
    pred_soft = torch.softmax(pred, dim=1)
    mask_pred = normalize(pred_soft) > conf_threshold
    conf_pseudo_p_w = pred_soft * mask_pred
    pseudo_label = torch.argmax(conf_pseudo_p_w.detach(), dim=1, keepdim=False)

    mask = (label == 1) | (label == 2)
    pseudo_mask = (pseudo_label == 1) | (pseudo_label == 2)
    pseudo_label[pseudo_mask] = 0
    pseudo_label[mask] = label[mask]
    return pseudo_label.detach()


if __name__ == "__main__":
    main()
