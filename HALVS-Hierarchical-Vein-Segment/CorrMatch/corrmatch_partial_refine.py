import argparse
import logging
import os
import pprint

from tqdm import tqdm
import numpy as np
import torch
from torch import nn
import torch.distributed as dist
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.optim import SGD
from torch.utils.data import DataLoader
from PIL import Image
import matplotlib
import matplotlib.pyplot as plt
from torchvision import transforms
from model.unet_de import UNet_LDMV2
from util.losses import DiceLoss,DiceLoss_refine
from util.classes import CLASSES

matplotlib.use('agg')
import yaml

from dataset.semi import SemiDataset
from model.semseg.deeplabv3plus import DeepLabV3Plus
from evaluate import evaluate
from util.ohem import ProbOhemCrossEntropy2d
from util.utils import count_params, init_log
from util.dist_helper import setup_distributed
from util.thresh_helper import ThreshController
from einops import rearrange
import random
from util.losses import exclusive_loss
# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

parser = argparse.ArgumentParser(description='Semi-Supervised Semantic Segmentation')
parser.add_argument('--config', type=str, required=True)
parser.add_argument('--labeled-id-path', type=str, required=True)
parser.add_argument('--unlabeled-id-path', type=str, required=True)
parser.add_argument('--partial_labeled_id_path',type=str, required=True)
parser.add_argument('--save-path', type=str, required=True)
parser.add_argument('--local-rank', default=0, type=int)
parser.add_argument('--port', default=None, type=int)


def init_seeds(seed=0, cuda_deterministic=False):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    cudnn.enabled = True
    # Speed-reproducibility tradeoff https://pytorch.org/docs/stable/notes/randomness.html
    if cuda_deterministic:  # slower, more reproducible
        cudnn.deterministic = True
        cudnn.benchmark = False
    else:  # faster, less reproducible
        cudnn.deterministic = False
        cudnn.benchmark = True


def main():
    args = parser.parse_args()

    cfg = yaml.load(open(args.config, "r"), Loader=yaml.Loader)

    logger = init_log('global', logging.INFO)
    logger.propagate = 0

    rank, word_size = setup_distributed(port=args.port)

    if rank == 0:
        logger.info('{}\n'.format(pprint.pformat(cfg)))

    if rank == 0:
        os.makedirs(args.save_path, exist_ok=True)
    init_seeds(0, False)

    model = DeepLabV3Plus(cfg)
    refine_model = UNet_LDMV2(in_chns=3+3, class_num=4, out_chns=4, ldm_method='replace', ldm_beta_sch='cosine', ts=10, ts_sample=2).cuda()
    # sam = sam_model_registry["vit_b"](checkpoint="sam/checkpoints/sam_vit_b.pth")
    # sam.cuda()
    criterion_refine_ce = nn.CrossEntropyLoss()
    refine_optimizer = SGD(refine_model.parameters(), lr=cfg['lr'], momentum=0.9, weight_decay=0.0001)
    criterion_dice = DiceLoss(n_classes=4)
    dice_refine = DiceLoss_refine(n_classes=4)
    if rank == 0:
        logger.info('Total params: {:.1f}M\n'.format(count_params(model)))

    optimizer = SGD([{'params': model.backbone.parameters(), 'lr': cfg['lr']},
                     {'params': [param for name, param in model.named_parameters() if 'backbone' not in name],
                      'lr': cfg['lr'] * cfg['lr_multi']}], lr=cfg['lr'], momentum=0.9, weight_decay=1e-4)

    local_rank = int(os.environ["LOCAL_RANK"])
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model.cuda()

    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank],
                                                      output_device=local_rank, find_unused_parameters=True)

    if cfg['criterion']['name'] == 'CELoss':
        criterion_l = nn.CrossEntropyLoss(**cfg['criterion']['kwargs']).cuda(local_rank)
    elif cfg['criterion']['name'] == 'OHEM':
        criterion_l = ProbOhemCrossEntropy2d(**cfg['criterion']['kwargs']).cuda(local_rank)
    else:
        raise NotImplementedError('%s criterion is not implemented' % cfg['criterion']['name'])

    criterion_u = nn.CrossEntropyLoss(reduction='none').cuda(local_rank)
    criterion_kl = nn.KLDivLoss(reduction='none').cuda(local_rank)

    trainset_u = SemiDataset(cfg['dataset'], cfg['data_root'], 'train_u',
                             cfg['crop_size'], args.unlabeled_id_path)
    trainset_l = SemiDataset(cfg['dataset'], cfg['data_root'], 'train_l',
                             cfg['crop_size'], args.labeled_id_path, nsample=len(trainset_u.ids))
    trainset_p = SemiDataset(cfg['dataset'], cfg['data_root'], 'train_p',
                             cfg['crop_size'], args.partial_labeled_id_path, nsample=len(trainset_u.ids))
    valset = SemiDataset(cfg['dataset'], cfg['data_root'], 'val')

    trainsampler_l = torch.utils.data.distributed.DistributedSampler(trainset_l)
    trainloader_l = DataLoader(trainset_l, batch_size=cfg['batch_size'],
                               pin_memory=False, num_workers=4, drop_last=True, sampler=trainsampler_l)
    trainsampler_u = torch.utils.data.distributed.DistributedSampler(trainset_u)
    trainloader_u = DataLoader(trainset_u, batch_size=cfg['batch_size'],
                               pin_memory=False, num_workers=4, drop_last=True, sampler=trainsampler_u)
    
    trainsampler_p = torch.utils.data.distributed.DistributedSampler(trainset_p)
    trainloader_p = DataLoader(trainset_p, batch_size=cfg['batch_size'],
                               pin_memory=True, num_workers=1, drop_last=True, sampler=trainsampler_p)
    valsampler = torch.utils.data.distributed.DistributedSampler(valset)
    valloader = DataLoader(valset, batch_size=1, pin_memory=True, num_workers=4,
                           drop_last=False, sampler=valsampler)

    total_iters = len(trainloader_u) * cfg['epochs']
    previous_best = 0.0
    thresh_controller = ThreshController(nclass=4, momentum=0.999, thresh_init=cfg['thresh_init'])
    previous_best = 0.0
    past_max_iou = 0.0


    color_map = {
            0: (0, 0, 0),    # background class
            1: (255, 0, 0),  # class 1 (red)
            2: (0, 255, 0),  # class 2 (green)
            3: (0, 0, 255),  # class 3 (blue)
        }
    for epoch in range(cfg['epochs']):
        if rank == 0:
            logger.info('===========> Epoch: {:}, LR: {:.4f}, Previous best: {:.2f}'.format(
                epoch, optimizer.param_groups[0]['lr'], previous_best))

        total_loss, total_loss_x, total_loss_s, total_loss_w_fp = 0.0, 0.0, 0.0, 0.0
        total_loss_kl = 0.0
        total_loss_corr_ce, total_loss_corr_u = 0.0, 0.0
        total_mask_ratio = 0.0

        trainloader_l.sampler.set_epoch(epoch)
        trainloader_u.sampler.set_epoch(epoch)
        trainloader_p.sampler.set_epoch(epoch)

        loader = zip(trainloader_l,trainloader_p, trainloader_u, trainloader_u)

        if rank == 0:
            tbar = tqdm(total=len(trainloader_l))

        for i, ((img_x, mask_x),
                (img_p_w,mask_p,img_p_s),
                (img_u_w, img_u_s1, _, ignore_mask, cutmix_box1, _),
                (img_u_w_mix, img_u_s1_mix, _, ignore_mask_mix, _, _)) in enumerate(loader):

            img_x, mask_x = img_x.cuda(), mask_x.cuda()
            img_u_w = img_u_w.cuda()
            img_u_s1, ignore_mask = img_u_s1.cuda(), ignore_mask.cuda()
            cutmix_box1 = cutmix_box1.cuda()
            img_p_w,mask_p,img_p_s = img_p_w.cuda(),mask_p.cuda(),img_p_s.cuda()
            img_u_w_mix = img_u_w_mix.cuda()
            img_u_s1_mix = img_u_s1_mix.cuda()
            ignore_mask_mix = ignore_mask_mix.cuda()
            b, c, h, w = img_x.shape

            with torch.no_grad():
                model.eval()
                res_u_w_mix = model(img_u_w_mix, need_fp=False, use_corr=False)
                pred_u_w_mix = res_u_w_mix['out'].detach()
                conf_u_w_mix = pred_u_w_mix.softmax(dim=1).max(dim=1)[0]
                mask_u_w_mix = pred_u_w_mix.argmax(dim=1)
                img_u_s1[cutmix_box1.unsqueeze(1).expand(img_u_s1.shape) == 1] = \
                    img_u_s1_mix[cutmix_box1.unsqueeze(1).expand(img_u_s1.shape) == 1]
                pred_p_w = model(img_p_w)['out']
                # print(pred_p_w.shape)
                conf_p_w = pred_p_w.softmax(dim=1).max(dim=1)[0]#(B,H,W)
                label_p_w = pred_p_w.argmax(dim=1)#弱增强后的当作伪标签

            model.train()
            
            pred_p_s = model(img_p_s)['out']

            num_lb, num_ulb = img_x.shape[0], img_u_w.shape[0]

            res_w = model(torch.cat((img_x, img_u_w)), need_fp=True, use_corr=True)

            preds = res_w['out']
            preds_fp = res_w['out_fp']
            preds_corr = res_w['corr_out']
            preds_corr_map = res_w['corr_map'].detach()
            pred_x_corr, pred_u_w_corr = preds_corr.split([num_lb, num_ulb])
            pred_u_w_corr_map = preds_corr_map[num_lb:]
            pred_x, pred_u_w = preds.split([num_lb, num_ulb])
            pred_u_w_fp = preds_fp[num_lb:]
            pred_x_soft = torch.softmax(pred_x,dim=1) 
            pseudo_mask = (normalize(pred_x_soft)>cfg['conf_threshold'])
            pred_mask = pred_x_soft*pseudo_mask
            pred_x_output = torch.argmax(pred_mask.detach(),dim=1,keepdim=False)
            res_s = model(img_u_s1, need_fp=False, use_corr=True)
            pred_u_s1 = res_s['out']
            pred_u_s1_corr = res_s['corr_out']

            pred_u_w = pred_u_w.detach()
            conf_u_w = pred_u_w.detach().softmax(dim=1).max(dim=1)[0]
            mask_u_w = pred_u_w.detach().argmax(dim=1)

            mask_u_w_cutmixed1, conf_u_w_cutmixed1, ignore_mask_cutmixed1 = \
                mask_u_w.clone(), conf_u_w.clone(), ignore_mask.clone()
            corr_map_u_w_cutmixed1 = pred_u_w_corr_map.clone()
            b_sample, c_sample, _, _ = corr_map_u_w_cutmixed1.shape

            cutmix_box1_map = (cutmix_box1 == 1)

            mask_u_w_cutmixed1[cutmix_box1_map] = mask_u_w_mix[cutmix_box1_map]
            mask_u_w_cutmixed1_copy = mask_u_w_cutmixed1.clone()
            conf_u_w_cutmixed1[cutmix_box1_map] = conf_u_w_mix[cutmix_box1_map]
            ignore_mask_cutmixed1[cutmix_box1_map] = ignore_mask_mix[cutmix_box1_map]
            cutmix_box1_sample = rearrange(cutmix_box1_map, 'n h w -> n 1 h w')
            ignore_mask_cutmixed1_sample = rearrange((ignore_mask_cutmixed1 != 255), 'n h w -> n 1 h w')
            corr_map_u_w_cutmixed1 = (corr_map_u_w_cutmixed1 * ~cutmix_box1_sample * ignore_mask_cutmixed1_sample).bool()

            thresh_controller.thresh_update(pred_u_w.detach(), ignore_mask_cutmixed1, update_g=True)
            thresh_global = thresh_controller.get_thresh_global()
            
            pred_u_s_soft = pred_u_s1.softmax(dim=1)
            pseudo_strong = (normalize(pred_u_s_soft)>cfg["conf_threshold"])
            outputs_strong_masked = pred_u_s_soft * pseudo_strong
            pseudo_outputs_strong = torch.argmax(outputs_strong_masked.detach(), dim=1, keepdim=False)

            mask_u_w_cutmixed_forref = mask_u_w_cutmixed1 * (conf_u_w_cutmixed1 >= cfg["conf_threshold"])
            conf_fliter_u_w = ((conf_u_w_cutmixed1 >= thresh_global) & (ignore_mask_cutmixed1 != 255))
            conf_fliter_u_w_without_cutmix = conf_fliter_u_w.clone()
            conf_fliter_u_w_sample = rearrange(conf_fliter_u_w_without_cutmix, 'n h w -> n 1 h w')

            segments = (corr_map_u_w_cutmixed1 * conf_fliter_u_w_sample).bool()
            
            
            mask_p_label = (mask_p == 1) | (mask_p == 2)#bool变量矩阵
            mask_p_unlabel = ~mask_p_label#标签中0的索引
            mask_p_unlabel_high_conf = mask_p_unlabel & ((label_p_w == 0) | (label_p_w == 3)) &\
                                    (conf_p_w >= cfg["conf_threshold"])#预测成0，3类别且高置信度>0.85
            mask_p_unlabel_others = mask_p_unlabel ^ mask_p_unlabel_high_conf
            # criterion for supervised loss
            criterion_partial = torch.nn.CrossEntropyLoss(reduction='none',ignore_index=255).cuda(local_rank)
                           
            # supervised loss for label 1 and 2
            
            partial_sup_loss = criterion_partial(pred_p_s, mask_p)
            partial_sup_loss = partial_sup_loss * mask_p_label
            partial_sup_loss = partial_sup_loss.sum() / (mask_p_label.sum() + 1)

            # loss for label 0 and 3 with high confidence

            partial_unsup_loss_1 = criterion_partial(pred_p_s, label_p_w)
            partial_unsup_loss_1 = partial_unsup_loss_1 * mask_p_unlabel_high_conf
            partial_unsup_loss_1 = partial_unsup_loss_1.sum() / (mask_p_unlabel_high_conf.sum() + 1)

            # exclusive loss for other pixels(we don't want these pixels predicted as label 1 and 2)
            
            partial_unsup_loss_2 = exclusive_loss(pred_p_s, exclude_label=[0 ,1, 2])
            partial_unsup_loss_2 = partial_unsup_loss_2 * mask_p_unlabel_others
            partial_unsup_loss_2 = partial_unsup_loss_2.sum() / (mask_p_unlabel_others.sum() + 1)
            
            
            loss_p = partial_sup_loss + partial_unsup_loss_1 + partial_unsup_loss_2

            for img_idx in range(b_sample):
                for segment_idx in range(c_sample):

                    segment = segments[img_idx, segment_idx]
                    segment_ori = corr_map_u_w_cutmixed1[img_idx, segment_idx]
                    high_conf_ratio = torch.sum(segment)/torch.sum(segment_ori)
                    if torch.sum(segment) == 0 or high_conf_ratio < thresh_global:
                        continue
                    unique_cls, count = torch.unique(mask_u_w_cutmixed1[img_idx][segment==1], return_counts=True)

                    if torch.max(count) / torch.sum(count) > thresh_global:
                        top_class = unique_cls[torch.argmax(count)]
                        mask_u_w_cutmixed1[img_idx][segment_ori==1] = top_class
                        conf_fliter_u_w_without_cutmix[img_idx] = conf_fliter_u_w_without_cutmix[img_idx] | segment_ori
            conf_fliter_u_w_without_cutmix = conf_fliter_u_w_without_cutmix | conf_fliter_u_w


            loss_x = criterion_l(pred_x, mask_x)
            loss_x_corr = criterion_l(pred_x_corr, mask_x)

            loss_u_s1 = criterion_u(pred_u_s1, mask_u_w_cutmixed1)
            loss_u_s1 = loss_u_s1 * conf_fliter_u_w_without_cutmix
            loss_u_s1 = torch.sum(loss_u_s1) / torch.sum(ignore_mask_cutmixed1 != 255).item()

            loss_u_corr_s1 = criterion_u(pred_u_s1_corr, mask_u_w_cutmixed1)
            loss_u_corr_s1 = loss_u_corr_s1 * conf_fliter_u_w_without_cutmix
            loss_u_corr_s1 = torch.sum(loss_u_corr_s1) / torch.sum(ignore_mask_cutmixed1 != 255).item()
            loss_u_corr_s = loss_u_corr_s1

            loss_u_corr_w = criterion_u(pred_u_w_corr, mask_u_w)
            loss_u_corr_w = loss_u_corr_w * ((conf_u_w >= thresh_global) & (ignore_mask != 255))
            loss_u_corr_w = torch.sum(loss_u_corr_w) / torch.sum(ignore_mask != 255).item()
            loss_u_corr = 0.5 * (loss_u_corr_s + loss_u_corr_w)

            softmax_pred_u_w = F.softmax(pred_u_w.detach(), dim=1)
            logsoftmax_pred_u_s1 = F.log_softmax(pred_u_s1, dim=1)

            loss_u_kl_sa2wa = criterion_kl(logsoftmax_pred_u_s1, softmax_pred_u_w)
            loss_u_kl_sa2wa = torch.sum(loss_u_kl_sa2wa, dim=1) * conf_fliter_u_w
            loss_u_kl_sa2wa = torch.sum(loss_u_kl_sa2wa) / torch.sum(ignore_mask_cutmixed1 != 255).item()
            loss_u_kl = loss_u_kl_sa2wa

            loss_u_w_fp = criterion_u(pred_u_w_fp, mask_u_w)
            loss_u_w_fp = loss_u_w_fp * ((conf_u_w >= thresh_global) & (ignore_mask != 255))
            loss_u_w_fp = torch.sum(loss_u_w_fp) / torch.sum(ignore_mask != 255).item()

            loss_u_p = ( 0.5 * loss_x + 0.5 * loss_x_corr + loss_u_s1 * 0.25 + loss_u_kl * 0.25 + loss_u_w_fp * 0.25 + 0.25 * loss_u_corr) / 2.0
            loss = loss_u_p + cfg['lambda'] * loss_p
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            pseudo_outputs_for_refine = pred_x_output.detach().clone()  # lab+unlab
            pseudo_outputs_numpy = pseudo_outputs_for_refine.clone().detach().cpu().numpy()
            pseudo_outputs_color = pl_weak_embed(color_map, pseudo_outputs_numpy)
            
            pseudo_outputs_for_refine_mix = mask_u_w_cutmixed_forref.detach().clone()
            pseudo_outputs_numpy_mix =  pseudo_outputs_for_refine_mix.clone().detach().cpu().numpy()
            pseudo_outputs_for_refine_mix_color = pl_weak_embed(color_map, pseudo_outputs_numpy_mix)

            pseudo_outputs_strong_forrefine = pseudo_outputs_strong.detach().clone()
            pseudo_outputs_strong_numpy = pseudo_outputs_strong_forrefine.cpu().numpy()
            pseudo_outputs_strong_color = pl_strong_embed(color_map, pseudo_outputs_strong_numpy)
            label_batch_numpy = mask_x.cpu().numpy()
            label_batch_color = label_embed(color_map, label_batch_numpy)
            label_batch_color = torch.cat(
                (label_batch_color.cuda(), pseudo_outputs_color[num_lb:].cuda()), dim=0
            )  # lab+unlab
            # 计算 Dice 损失
            # print(mask_x.size())
            t = dice_refine(
                inputs=pseudo_outputs_for_refine.unsqueeze(1),
                target=mask_x.unsqueeze(1),
                oh_input = True,
                weight = [0,0.2,0.2,0.6]
            )

            t = torch.ones((pseudo_outputs_color.shape[0]), dtype=torch.float32, device='cuda') * t * 999 * 4
           
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
            sup_loss_cedice = (
                0.5*criterion_l(ref_outputs[:num_lb], mask_x.long()) 
                + 0.5*criterion_dice(ref_outputs_soft[:num_lb], mask_x.unsqueeze(1).float())
            )
            sup_loss_ref = sup_loss_cedice + lat_loss_sup
            # 使用精炼后的输出生成新的伪标签
            ref_soft = ref_outputs_soft
            # ref_pseudo_mask = (normalize(ref_soft) > cfg['conf_thresh']).float()
            # ref_outputs_masked = ref_soft * ref_pseudo_mask
            ref_pseudo_outputs = pseudo_outputs_for_refine_mix  # lab+unlab

            # 潜在上下文精炼模块 - 无标签数据
            t2 = dice_refine(
                pseudo_outputs_strong_forrefine.unsqueeze(1),
                ref_pseudo_outputs.unsqueeze(1),
                oh_input=True,
                weight = [0,0.2,0.2,0.6]
            )

            t2 = torch.ones((pseudo_outputs_strong_color.shape[0]), dtype=torch.float32, device='cuda') * t2 * 999 * 4

            lat_loss_unsup, ref_outputs_strong = refine_model(
                pseudo_outputs_strong_color.cuda(),
                t2,
                img_u_s1.cuda(),
                training=True,
                good=pseudo_outputs_for_refine_mix_color.cuda(),
                )

            ref_outputs_strong_soft = torch.softmax(ref_outputs_strong, dim=1)

            # 计算复合损失
            # ref_comp_loss, ref_as_weight = get_comp_loss(weak=ref_soft, strong=ref_outputs_strong_soft)
            unsup_loss_cedice = (
                0.5*criterion_refine_ce(ref_outputs_strong, ref_pseudo_outputs)
                + 0.5*criterion_dice(ref_outputs_strong_soft, ref_pseudo_outputs.unsqueeze(1))
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
            refine_loss = sup_loss_ref +  unsup_loss_ref + partial_loss_ref
            # 优化精炼模型
            refine_optimizer.zero_grad()
            refine_loss.backward()
            refine_optimizer.step()
            
        # 3. 模型校正损失
            if epoch > 10:
                # 重新计算 t
                t = dice_refine(
                    pseudo_outputs_for_refine[:num_lb].unsqueeze(1),
                    mask_x.unsqueeze(1),
                    oh_input=True,
                    weight = [0,0.2,0.2,0.6]
                )
                t = torch.ones((pseudo_outputs_color.shape[0]), dtype=torch.float32, device='cuda') * t * 999 * 4

                # 精炼模型推理
                ref_outputs = refine_model(
                    pseudo_outputs_color.cuda(),
                    t,
                    img_x.cuda(),
                    training=False,
                )

                ref_outputs_soft_for_refine = torch.softmax(ref_outputs, dim=1)
                pseudo_mask = (normalize(ref_outputs_soft_for_refine) > cfg['conf_threshold']).float()
                ref_outputs_soft_masked = ref_outputs_soft_for_refine * pseudo_mask
                pseudo_outputs_ref = torch.argmax(ref_outputs_soft_masked.detach(), dim=1, keepdim=False)

                # 计算校正损失
                pred_x_updated=model(img_x)['out']
                pred_x_updated = pred_x_updated[:num_lb]
                loss_rect = (
                   0.5*criterion_refine_ce(pred_x_updated, pseudo_outputs_ref[:num_lb]) 
                    + 0.5*criterion_dice(pred_x_updated.softmax(dim=1), pseudo_outputs_ref[:num_lb].unsqueeze(1).float())
                )

                # 优化分割模型
                optimizer.zero_grad()
                loss_rect.backward()
                optimizer.step()

            total_loss += loss.item()
            total_loss_x += loss_x.item()
            total_loss_s += loss_u_s1.item()
            total_loss_kl += loss_u_kl.item()
            total_loss_w_fp += loss_u_w_fp.item()
            total_loss_corr_ce += loss_x_corr.item()
            total_loss_corr_u += loss_u_corr.item()
            total_mask_ratio += ((conf_u_w >= thresh_global) & (ignore_mask != 255)).sum().item() / \
                                (ignore_mask != 255).sum().item()

            iters = epoch * len(trainloader_u) + i
            lr = cfg['lr'] * (1 - iters / total_iters) ** 0.9
            optimizer.param_groups[0]["lr"] = lr
            optimizer.param_groups[1]["lr"] = lr * cfg['lr_multi']

            if rank == 0:
                tbar.set_description(' Total loss: {:.3f}, Loss x: {:.3f}, loss_corr_ce: {:.3f} '
                                     'Loss s: {:.3f}, Loss w_fp: {:.3f},  Mask: {:.3f}, loss_corr_u: {:.3f}'.format(
                    total_loss / (i + 1), total_loss_x / (i + 1), total_loss_corr_ce / (i + 1), total_loss_s / (i + 1),
                    total_loss_w_fp / (i + 1), total_mask_ratio / (i + 1), total_loss_corr_u / (i + 1)))
                tbar.update(1)

        if rank == 0:
            tbar.close()

        if cfg['dataset'] == 'cityscapes':
            eval_mode = 'center_crop' if epoch < cfg['epochs'] - 20 else 'sliding_window'
        else:
            eval_mode = 'original'
        torch.cuda.empty_cache()
        res_val = evaluate(model, valloader, eval_mode, cfg)
        mIOU = res_val['mIOU']
        class_IOU = res_val['iou_class']
        mIOU_without_bg = res_val['mIOU_without_bg']
        iou_class_4 = class_IOU[3]
        max_iou = max(iou_class_4,past_max_iou)
        past_max_iou = max_iou
        torch.distributed.barrier()

        if rank == 0:
         for (cls_idx, iou) in enumerate(class_IOU):
                logger.info('***** Evaluation ***** >>>> Class [{:} {:}] '
                            'IoU: {:.2f}'.format(cls_idx, CLASSES[cfg['dataset']][cls_idx], iou * 100))
        logger.info('***** Evaluation {} ***** >>>> meanIOU: {:.4f} \n'.format(eval_mode, mIOU))
        logger.info('***** Evaluation {} ***** >>>> meanIOU without bg : \n{}\n'.format(eval_mode,mIOU_without_bg))
        logger.info('***** Evaluation {} ***** >>>> best_3rd_IoU: \n{}\n'.format(eval_mode,past_max_iou))

        if mIOU_without_bg > previous_best and rank == 0:
            if previous_best != 0:
                os.remove(os.path.join(args.save_path, '%s_%.3f.pth' % (cfg['backbone'], previous_best)))
            previous_best = mIOU_without_bg
            torch.save(model.module.state_dict(), os.path.join(args.save_path, '%s_%.3f.pth' % (cfg['backbone'], mIOU_without_bg)))
        torch.distributed.barrier()
        torch.cuda.empty_cache()
        
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


if __name__ == '__main__':
    main()
