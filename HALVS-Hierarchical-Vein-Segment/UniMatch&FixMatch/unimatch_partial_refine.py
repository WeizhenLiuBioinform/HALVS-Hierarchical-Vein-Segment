import argparse
import logging
import os
import pprint
import time

import torch
from torch import nn
import torch.backends.cudnn as cudnn
from torch.optim import SGD
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import yaml
from torch.nn import functional as F
from model.unet_de import UNet_LDMV2
from util.loss import DiceLoss,DiceLoss_refine
from dataset.semi import SemiDataset
from model.semseg.deeplabv3plus import DeepLabV3Plus
from supervised import evaluate
from util.classes import CLASSES
from util.ohem import ProbOhemCrossEntropy2d
from util.utils import count_params, init_log, AverageMeter
from util.dist_helper import setup_distributed,exclusive_loss
# from model.semseg.vision_transformer import SwinUnet
from PIL import Image
from torchvision import transforms
import numpy as np  

parser = argparse.ArgumentParser(description='Revisiting Weak-to-Strong Consistency in Semi-Supervised Semantic Segmentation')
parser.add_argument('--config', type=str, required=True)
parser.add_argument('--labeled-id-path', type=str, required=True)
parser.add_argument('--unlabeled-id-path', type=str, required=True)
parser.add_argument('--partial_labeled_id_path',type=str)
parser.add_argument('--save-path', type=str, required=True)
parser.add_argument('--local_rank', default=0, type=int)
parser.add_argument('--port', default=None, type=int)
# parser = argparse.ArgumentParser(description='Revisiting Weak-to-Strong Consistency in Semi-Supervised Semantic Segmentation')
# parser.add_argument('--config',default='/home/lia/UniMatch/configs/leaf_new.yaml',type=str)
# parser.add_argument('--labeled-id-path', default='/home/lia/UniMatch/splits/leaf_new/6_6_60/labeled.txt',type=str)
# parser.add_argument('--unlabeled-id-path', default='/home/lia/UniMatch/splits/leaf_new/6_6_60/unlabeled.txt',type=str)
# parser.add_argument('--partial_labeled_id_path', default='/home/lia/UniMatch/splits/leaf_new/6_6_60/partial_labeled.txt',type=str)
# parser.add_argument('--save-path', default='/home/lia/ssl/UniMatch/save',type=str)
# parser.add_argument('--local_rank', default=0, type=int)
# parser.add_argument('--port', default=25, type=int)


def main():
    i=0
    args = parser.parse_args()
    print(args)
    
    cfg = yaml.load(open(args.config, "r"), Loader=yaml.Loader)

    logger = init_log('global', logging.INFO)
    logger.propagate = 0

    rank, world_size = setup_distributed(port=args.port)

    if rank == 0:
        all_args = {**cfg, **vars(args), 'ngpus': world_size}
        logger.info('{}\n'.format(pprint.pformat(all_args)))
        
        writer = SummaryWriter(args.save_path)
        
        os.makedirs(args.save_path, exist_ok=True)
    
    cudnn.enabled = True
    cudnn.benchmark = True

    model = DeepLabV3Plus(cfg)
    optimizer = SGD([{'params': model.backbone.parameters(), 'lr': cfg['lr']},
                    {'params': [param for name, param in model.named_parameters() if 'backbone' not in name],
                    'lr': cfg['lr'] * cfg['lr_multi']}], lr=cfg['lr'], momentum=0.9, weight_decay=1e-4)
    
    if rank == 0:
        logger.info('Total params: {:.1f}M\n'.format(count_params(model)))

    local_rank = int(os.environ["LOCAL_RANK"])
    
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model.cuda()
    
    
    criterion_refine_ce = nn.CrossEntropyLoss()

    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], broadcast_buffers=False,
                                                      output_device=local_rank, find_unused_parameters=False)
    refine_model = UNet_LDMV2(in_chns=3+3, class_num=4, out_chns=4, ldm_method='replace', ldm_beta_sch='cosine', ts=10, ts_sample=2).cuda()
    refine_optimizer = SGD(refine_model.parameters(), lr=cfg['lr'], momentum=0.9, weight_decay=0.0001)

    if cfg['criterion']['name'] == 'CELoss':
        criterion_l = nn.CrossEntropyLoss(**cfg['criterion']['kwargs']).cuda(local_rank)
    elif cfg['criterion']['name'] == 'OHEM':
        criterion_l = ProbOhemCrossEntropy2d(**cfg['criterion']['kwargs']).cuda(local_rank)
    else:
        raise NotImplementedError('%s criterion is not implemented' % cfg['criterion']['name'])

    criterion_u = nn.CrossEntropyLoss(reduction='none').cuda(local_rank)
    criterion_dice = DiceLoss(n_classes=4)

    trainset_u = SemiDataset(cfg['dataset'], cfg['data_root'], 'train_u',
                             cfg['crop_size'], args.unlabeled_id_path)
    trainset_l = SemiDataset(cfg['dataset'], cfg['data_root'], 'train_l',
                             cfg['crop_size'], args.labeled_id_path, nsample=len(trainset_u.ids))
    #实例数据集trainset_u.ids=10000，迭代满10000次
    trainset_p = SemiDataset(cfg['dataset'], cfg['data_root'], 'train_p',
                             cfg['crop_size'], args.partial_labeled_id_path, nsample=len(trainset_u.ids))
    #
    valset = SemiDataset(cfg['dataset'], cfg['data_root'], 'val')
    dice_refine = DiceLoss_refine(n_classes=4)

    print(len(trainset_l.ids))
    print(len(valset.ids))
    trainsampler_l = torch.utils.data.distributed.DistributedSampler(trainset_l)
    trainloader_l = DataLoader(trainset_l, batch_size=cfg['batch_size'],
                               pin_memory=True, num_workers=1, drop_last=True, sampler=trainsampler_l)

    trainsampler_u = torch.utils.data.distributed.DistributedSampler(trainset_u)
    trainloader_u = DataLoader(trainset_u, batch_size=cfg['batch_size'],
                               pin_memory=True, num_workers=1, drop_last=True, sampler=trainsampler_u)

    #定义loader迭代器
    trainsampler_p = torch.utils.data.distributed.DistributedSampler(trainset_p)
    trainloader_p = DataLoader(trainset_p, batch_size=cfg['batch_size'],
                               pin_memory=True, num_workers=0, drop_last=True, sampler=trainsampler_p)



    valsampler = torch.utils.data.distributed.DistributedSampler(valset)
    valloader = DataLoader(valset, batch_size=2, pin_memory=True, num_workers=1,
                           drop_last=False, sampler=valsampler)

    

    total_iters = len(trainloader_u) * cfg['epochs']
    previous_best = 0.0
    past_max_iou = 0.0
    epoch = -1
    

    # checkpoint = torch.load('/home/lia/UniMatch/test/pth/soybean_20_20_200_pretrain')
    # # checkpoint = torch.load(os.path.join(args.save_path, 'latest.pth'))
    # model.load_state_dict(checkpoint['model'])
    # optimizer.load_state_dict(checkpoint['optimizer'])
    # epoch = checkpoint['epoch']
    # previous_best = checkpoint['previous_best']
    
    # if rank == 0:
    #     logger.info('************ Load from checkpoint at epoch %i\n' % epoch)

    if os.path.exists(os.path.join(args.save_path, 'latest.pth')):
        # checkpoint = torch.load('/home/lia/UniMatch/test/pth/soybean_20_20_200_pretrain')
        checkpoint = torch.load(os.path.join(args.save_path, 'latest.pth'))
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        epoch = checkpoint['epoch']
        previous_best = checkpoint['previous_best']
        
        if rank == 0:
            logger.info('************ Load from checkpoint at epoch %i\n' % epoch)
    ######
    for epoch in range(epoch + 1, cfg['epochs']):
        if rank == 0:
            logger.info('===========> Epoch: {:}, LR: {:.5f}, Previous best: {:.2f}'.format(
                epoch, optimizer.param_groups[0]['lr'], previous_best))

        total_loss = AverageMeter()#总loss
        #非partial部分
        total_loss_u_p = AverageMeter()
        total_loss_x = AverageMeter()#全监督部分的loss
        total_loss_s = AverageMeter()#强增强的两个loss
        total_loss_w_fp = AverageMeter()#drop后的loss
        #partial部分
        total_loss_p = AverageMeter()
        total_loss_p_sup_loss = AverageMeter()
        total_loss_p_unsup_loss_1 = AverageMeter()
        total_loss_p_unsup_loss_2 = AverageMeter()
        
        total_mask_ratio = AverageMeter()#

        trainloader_l.sampler.set_epoch(epoch)
        trainloader_u.sampler.set_epoch(epoch)
        trainloader_p.sampler.set_epoch(epoch)


        color_map = {
            0: (0, 0, 0),    # background class
            1: (255, 0, 0),  # class 1 (red)
            2: (0, 255, 0),  # class 2 (green)
            3: (0, 0, 255),  # class 3 (blue)
        }

        

        loader = zip(trainloader_l,trainloader_p,trainloader_u, trainloader_u)
        #从全标签、无标签、无标签中取
        # 其中全标签集合取 数据和标签（normalize、弱增强后的）
        # 无标签集合取 原图（弱增强）原图s1（弱增强后强增强）原图s2（弱增强后强增强），全0的ignore_mask,随机进行cutmix的s1，随机进行cutmix的s2
        #无标签集合取 
        for i, ((img_x, mask_x),
                (img_p_w,mask_p,img_p_s),#从partial迭代器中取弱增强和强增强
                (img_u_w, img_u_s1, img_u_s2, ignore_mask, cutmix_box1, cutmix_box2),
                (img_u_w_mix, img_u_s1_mix, img_u_s2_mix, ignore_mask_mix, _, _)) in enumerate(loader):#
            
            
            img_x, mask_x = img_x.cuda(), mask_x.cuda()

            img_p_w,mask_p,img_p_s = img_p_w.cuda(),mask_p.cuda(),img_p_s.cuda()

            img_u_w = img_u_w.cuda()

            img_u_s1, img_u_s2, ignore_mask = img_u_s1.cuda(), img_u_s2.cuda(), ignore_mask.cuda()

            cutmix_box1, cutmix_box2 = cutmix_box1.cuda(), cutmix_box2.cuda()
            img_u_w_mix = img_u_w_mix.cuda()
            img_u_s1_mix, img_u_s2_mix = img_u_s1_mix.cuda(), img_u_s2_mix.cuda()
            ignore_mask_mix = ignore_mask_mix.cuda()

            with torch.no_grad():
                model.eval()
                #无标签部分的伪标签生成
                pred_u_w_mix = model(img_u_w_mix)#2,4,321,321
                conf_u_w_mix = pred_u_w_mix.softmax(dim=1).max(dim=1)[0]#取类别通道上做softmax，并取下概率最大的，形状变成2，321，321
                mask_u_w_mix = pred_u_w_mix.argmax(dim=1)

                #partial部分的伪标签生成
                pred_p_w = model(img_p_w)
                conf_p_w = pred_p_w.softmax(dim=1).max(dim=1)[0]#(B,H,W)
                label_p_w = pred_p_w.argmax(dim=1)#弱增强后的当作伪标签

            img_u_s1[cutmix_box1.unsqueeze(1).expand(img_u_s1.shape) == 1] = \
                img_u_s1_mix[cutmix_box1.unsqueeze(1).expand(img_u_s1.shape) == 1]
            img_u_s2[cutmix_box2.unsqueeze(1).expand(img_u_s2.shape) == 1] = \
                img_u_s2_mix[cutmix_box2.unsqueeze(1).expand(img_u_s2.shape) == 1]

            model.train()

            #####实现cutmix增强，用于将两张输入图像进行拼接。等于说是img_u_s1和img_u_s1_mix拼在一起
            pred_p_s = model(img_p_s)

            num_lb, num_ulb = img_x.shape[0], img_u_w.shape[0]#2,2
            
            preds, preds_fp = model(torch.cat((img_x, img_u_w)), True)#preds代表原图（有标签）打入model后的输出，preds_fp代表原图弱增强 并且在编码后结合了FP扰动 再打入model后的输出。形状[4,4,321,321]
            pred_x, pred_u_w = preds.split([num_lb, num_ulb])#2，4，321，321 pred_x代表原图img_x经过网络的输出
            pred_u_w_fp = preds_fp[num_lb:]


            pred_u_s1, pred_u_s2 = model(torch.cat((img_u_s1, img_u_s2))).chunk(2)#pred_u_s1代表强增强的图片经过模型后的输出
            pred_u_s1, pred_u_s2 = model(torch.cat((img_u_s1, img_u_s2))).chunk(2)#pred_u_s1代表强增强的图片经过模型后的输出
            
            #partial强增强的图片经过模型后的输出
            
            #5个预测pred_x，pred_u_w，pred_u_w_fp，pred_u_s1，pred_u_s2

            pred_u_w = pred_u_w.detach()
            conf_u_w = pred_u_w.softmax(dim=1).max(dim=1)[0]#
            mask_u_w = pred_u_w.argmax(dim=1)#标签！也就是论文中的伪标签

            mask_u_w_cutmixed1, conf_u_w_cutmixed1, ignore_mask_cutmixed1 = \
                mask_u_w.clone(), conf_u_w.clone(), ignore_mask.clone()
            mask_u_w_cutmixed2, conf_u_w_cutmixed2, ignore_mask_cutmixed2 = \
                mask_u_w.clone(), conf_u_w.clone(), ignore_mask.clone()

            mask_u_w_cutmixed1[cutmix_box1 == 1] = mask_u_w_mix[cutmix_box1 == 1]
            conf_u_w_cutmixed1[cutmix_box1 == 1] = conf_u_w_mix[cutmix_box1 == 1]
            ignore_mask_cutmixed1[cutmix_box1 == 1] = ignore_mask_mix[cutmix_box1 == 1]

            mask_u_w_cutmixed2[cutmix_box2 == 1] = mask_u_w_mix[cutmix_box2 == 1]
            conf_u_w_cutmixed2[cutmix_box2 == 1] = conf_u_w_mix[cutmix_box2 == 1]
            ignore_mask_cutmixed2[cutmix_box2 == 1] = ignore_mask_mix[cutmix_box2 == 1]

            loss_x = criterion_l(pred_x, mask_x)####有监督部分的loss

            loss_u_s1 = criterion_u(pred_u_s1, mask_u_w_cutmixed1)
            loss_u_s1 = loss_u_s1 * ((conf_u_w_cutmixed1 >= cfg['conf_thresh']) & (ignore_mask_cutmixed1 != 255))
            loss_u_s1 = loss_u_s1.sum() / (ignore_mask_cutmixed1 != 255).sum().item()

            loss_u_s2 = criterion_u(pred_u_s2, mask_u_w_cutmixed2)
            loss_u_s2 = loss_u_s2 * ((conf_u_w_cutmixed2 >= cfg['conf_thresh']) & (ignore_mask_cutmixed2 != 255))
            loss_u_s2 = loss_u_s2.sum() / (ignore_mask_cutmixed2 != 255).sum().item()

            loss_u_w_fp = criterion_u(pred_u_w_fp, mask_u_w)
            loss_u_w_fp = loss_u_w_fp * ((conf_u_w >= cfg['conf_thresh']) & (ignore_mask != 255))
            loss_u_w_fp = loss_u_w_fp.sum() / (ignore_mask != 255).sum().item()

            loss_u_p = loss_x + loss_u_s1 * 0.25 + loss_u_s2 * 0.25 + loss_u_w_fp * 0.5

            # partial
            mask_p_label = (mask_p == 1) | (mask_p == 2)#bool变量矩阵
            mask_p_unlabel = ~mask_p_label#标签中0的索引
            mask_p_unlabel_high_conf = mask_p_unlabel & ((label_p_w == 0) | (label_p_w == 3)) &\
                                     (conf_p_w >= cfg["conf_threshold"])#预测成0，3类别且高置信度>0.85
            


            mask_p_unlabel_others = mask_p_unlabel ^ mask_p_unlabel_high_conf
            # criterion for supervised loss
            criterion_partial = torch.nn.CrossEntropyLoss(reduction='none',ignore_index=255).cuda(local_rank)
                           
            # supervised loss for label 1 and 2
            partial_sup_loss = criterion_partial(pred_p_s, mask_p)  # [batch, 256, 256]
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
            

            loss_p = partial_sup_loss +  partial_unsup_loss_1 + partial_unsup_loss_2

            
            
            loss = loss_u_p +  cfg['lambda'] * loss_p 
            torch.distributed.barrier()


            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            pseudo_outputs_for_refine = pred_x.detach().clone()  # 预测的输出
            pseudo_outputs_numpy = pseudo_outputs_for_refine.argmax(dim=1, keepdim=False).cpu().numpy()
            pseudo_outputs_color = pl_weak_embed(color_map, pseudo_outputs_numpy)

            # 准备真实标签的颜色编码
            label_batch_numpy = mask_x.cpu().numpy()
            label_batch_color = label_embed(color_map, label_batch_numpy)

            # 计算 Dice 系数
            t = dice_refine(
                inputs=pseudo_outputs_for_refine.argmax(dim=1, keepdim=True),
                target=mask_x.unsqueeze(1),
                oh_input=True,
                weight=[0,0.2,0.2,0.6]
            )
            t = torch.ones((pseudo_outputs_color.shape[0]), dtype=torch.float32, device='cuda') * t * 999 * 4

            # 精炼模型前向传播 - 有标签数据
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
                0.5 * criterion_l(ref_outputs, mask_x.long()) +
                0.5 * criterion_dice(ref_outputs_soft, mask_x.unsqueeze(1).float())
            )
            sup_loss_ref = sup_loss_cedice + lat_loss_sup

        # 准备伪标签 - 无标签数据
            pseudo_outputs_for_refine_mix1 = mask_u_w_cutmixed1.clone()
            pseudo_outputs_for_refine_mix2 = mask_u_w_cutmixed2.clone()
            pseudo_outputs_for_refine_mix = torch.cat((pseudo_outputs_for_refine_mix1, pseudo_outputs_for_refine_mix2), dim=0)
            pseudo_outputs_numpy_mix = pseudo_outputs_for_refine_mix.cpu().numpy()
            pseudo_outputs_for_refine_mix_color = pl_weak_embed(color_map, pseudo_outputs_numpy_mix)

            # 准备强增强预测
            pseudo_outputs_strong_for_refine1 = pred_u_s1.detach().clone().argmax(dim=1, keepdim=False)
            pseudo_outputs_strong_for_refine2 = pred_u_s2.detach().clone().argmax(dim=1, keepdim=False)
            pseudo_outputs_strong_for_refine = torch.cat((pseudo_outputs_strong_for_refine1, pseudo_outputs_strong_for_refine2), dim=0)
            pseudo_outputs_strong_numpy = pseudo_outputs_strong_for_refine.cpu().numpy()
            pseudo_outputs_strong_color = pl_strong_embed(color_map, pseudo_outputs_strong_numpy)

            # 计算 Dice 系数 - 无标签数据
            t2 = dice_refine(
                pseudo_outputs_strong_for_refine.unsqueeze(1),
                pseudo_outputs_for_refine_mix.unsqueeze(1),
                oh_input=True,
                weight=[0,0.2,0.2,0.6]
            )
            t2 = torch.ones((pseudo_outputs_strong_color.shape[0]), dtype=torch.float32, device='cuda') * t2 * 999 * 4

            # 精炼模型前向传播 - 无标签数据
            lat_loss_unsup, ref_outputs_strong = refine_model(
                pseudo_outputs_strong_color.cuda(),
                t2,
                torch.cat((img_u_s1, img_u_s2)).cuda(),
                training=True,
                good=pseudo_outputs_for_refine_mix_color.cuda(),
            )
            ref_outputs_strong_soft = torch.softmax(ref_outputs_strong, dim=1)

            # 计算无监督精炼损失
            ref_pseudo_outputs = pseudo_outputs_for_refine_mix  # 伪标签
            unsup_loss_cedice = (
                0.5 * criterion_refine_ce(ref_outputs_strong, ref_pseudo_outputs) +
                0.5 * criterion_dice(ref_outputs_strong_soft, ref_pseudo_outputs.unsqueeze(1))
            )
            unsup_loss_ref = unsup_loss_cedice + lat_loss_unsup
            
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

            # 总精炼损失
            refine_loss = sup_loss_ref + unsup_loss_ref + partial_loss_ref

            # 更新精炼模型
            refine_optimizer.zero_grad()
            refine_loss.backward()
            refine_optimizer.step()

            # 模型校正损失
            if epoch > 10:
                # 重新计算 t
                t = dice_refine(
                    pseudo_outputs_for_refine.argmax(dim=1, keepdim=True),
                    mask_x.unsqueeze(1),
                    oh_input=True,
                    weight=[0,0.2,0.2,0.6]
                )
                t = torch.ones((pseudo_outputs_color.shape[0]), dtype=torch.float32, device='cuda') * t * 999 * 4

                # 精炼模型推理
                ref_outputs = refine_model(
                    pseudo_outputs_color.cuda(),
                    t,
                    img_x.cuda(),
                    training=False,
                )
                ref_outputs_soft = torch.softmax(ref_outputs, dim=1)

                # 生成精炼后的伪标签
                pseudo_mask = (normalize(ref_outputs_soft) > cfg['conf_thresh']).float()
                ref_outputs_soft_masked = ref_outputs_soft * pseudo_mask
                pseudo_outputs_ref = ref_outputs_soft_masked.argmax(dim=1, keepdim=False)

                # 计算校正损失
                pred_x_updated = model(img_x)
                loss_rect = (
                    0.5 * criterion_refine_ce(pred_x_updated, pseudo_outputs_ref) +
                    0.5 * criterion_dice(pred_x_updated.softmax(dim=1), pseudo_outputs_ref.unsqueeze(1).float())
                )

                # 优化分割模型
                optimizer.zero_grad()
                loss_rect.backward()
                optimizer.step()

            total_loss.update(loss.item())

            total_loss_u_p.update(loss_u_p.item())
            total_loss_x.update(loss_x.item())
            total_loss_s.update((loss_u_s1.item() + loss_u_s2.item()) / 2.0)
            total_loss_w_fp.update(loss_u_w_fp.item())

            total_loss_p.update(loss_p.item())
            total_loss_p_sup_loss.update(partial_sup_loss.item())
            total_loss_p_unsup_loss_1.update(partial_unsup_loss_1.item())
            total_loss_p_unsup_loss_2.update(partial_unsup_loss_2.item())



            mask_ratio = ((conf_u_w >= cfg['conf_thresh']) & (ignore_mask != 255)).sum().item() / \
                (ignore_mask != 255).sum()

            total_mask_ratio.update(mask_ratio.item())

            iters = epoch * len(trainloader_u) + i
            lr = cfg['lr'] * (1 - iters / total_iters) ** 0.9
            optimizer.param_groups[0]["lr"] = lr
            optimizer.param_groups[1]["lr"] = lr * cfg['lr_multi']
            
            if rank == 0:
                writer.add_scalar('train/loss_all', loss.item(), iters)
                writer.add_scalar('train/loss_x', loss_x.item(), iters)
                writer.add_scalar('train/loss_s', (loss_u_s1.item() + loss_u_s2.item()) / 2.0, iters)
                writer.add_scalar('train/loss_w_fp', loss_u_w_fp.item(), iters)
                writer.add_scalar('train/mask_ratio', mask_ratio, iters)
                writer.add_scalar('train/loss_p', loss_p.item(), iters)
                writer.add_scalar('train/partial_sup_loss', partial_sup_loss.item(), iters)
                writer.add_scalar('train/partial_unsup_loss_1', partial_unsup_loss_1.item(), iters)
                writer.add_scalar('train/partial_unsup_loss_2', partial_unsup_loss_2.item(), iters)



            if (i % (len(trainloader_u) // 8) == 0) and (rank == 0):
                logger.info('Iters: {:}, Total loss: {:.3f},Partial loss: {:.3f},Partial sup_loss: {:.3f} ,Partial unsup_loss_1: {:.3f},Partial unsup_loss_2: {:.3f},Loss x: {:.3f}, Loss s: {:.3f}, Loss w_fp: {:.3f}, Mask ratio: '
                            '{:.3f}'.format(i, total_loss.avg, total_loss_p.avg,total_loss_p_sup_loss.avg,total_loss_p_unsup_loss_1.avg, total_loss_p_unsup_loss_2.avg,total_loss_x.avg, total_loss_s.avg,
                                            total_loss_w_fp.avg, total_mask_ratio.avg))

            # if (i % (len(trainloader_u) // 8) == 0) and (rank == 0):
            #         logger.info('Iters: {:}, Total loss: {:.3f},Partial loss: {:.3f},Partial sup_loss: {:.3f} ,Partial unsup_loss_2: {:.3f},Loss x: {:.3f}, Loss s: {:.3f}, Loss w_fp: {:.3f}, Mask ratio: '
            #                 '{:.3f}'.format(i, total_loss.avg, total_loss_p.avg,total_loss_p_sup_loss.avg,total_loss_p_unsup_loss_2.avg,total_loss_x.avg, total_loss_s.avg,
            #                                 total_loss_w_fp.avg, total_mask_ratio.avg))

            # if (i % (len(trainloader_u) // 8) == 0) and (rank == 0):
            #     logger.info('Iters: {:}, Total loss: {:.3f},Partial loss: {:.3f},Partial sup_loss: {:.3f} ,Partial unsup_loss_1: {:.3f},Loss x: {:.3f}, Loss s: {:.3f}, Loss w_fp: {:.3f}, Mask ratio: '
            #                 '{:.3f}'.format(i, total_loss.avg, total_loss_p.avg,total_loss_p_sup_loss.avg,total_loss_p_unsup_loss_1.avg, total_loss_x.avg, total_loss_s.avg,
            #                                 total_loss_w_fp.avg, total_mask_ratio.avg))
            # if (i % (len(trainloader_u) // 8) == 0) and (rank == 0):
            #     logger.info('Iters: {:}, Total loss: {:.3f},Partial loss: {:.3f},Partial sup_loss: {:.3f} ,Loss x: {:.3f}, Loss s: {:.3f}, Loss w_fp: {:.3f}, Mask ratio: '
            #                 '{:.3f}'.format(i, total_loss.avg, total_loss_p.avg,total_loss_p_sup_loss.avg,total_loss_x.avg, total_loss_s.avg,
            #                                 total_loss_w_fp.avg, total_mask_ratio.avg))



        eval_mode = 'sliding_window' if cfg['dataset'] == 'cityscapes' else 'original'
        mIoU, mIoU_without_bg, iou_class = evaluate(model, valloader, eval_mode, cfg)

        iou_class_4 = iou_class[3]
        max_iou = max(iou_class_4, past_max_iou)
        past_max_iou = max_iou
        if rank == 0:
            for (cls_idx, iou) in enumerate(iou_class):
                logger.info('***** Evaluation ***** >>>> Class [{:} {:}] '
                            'IoU: {:.2f}'.format(cls_idx, CLASSES[cfg['dataset']][cls_idx], iou))
            logger.info('***** Evaluation {} ***** >>>> MeanIoU: {:.2f}'.format(eval_mode, mIoU))
            logger.info('***** Evaluation {} ***** >>>> MeanIoU without bg: {:.2f}'.format(eval_mode, mIoU_without_bg))
            logger.info('***** Evaluation {} ***** >>>> best_3rd_IoU: {:.2f}\n'.format(eval_mode, past_max_iou))


            writer.add_scalar('eval/mIoU', mIoU, epoch)
            writer.add_scalar('eval/mIoU_without_bg', mIoU_without_bg, epoch)
            writer.add_scalar('eval/3rd_mIoU', past_max_iou, epoch)
            for i, iou in enumerate(iou_class):
                writer.add_scalar('eval/%s_IoU' % (CLASSES[cfg['dataset']][i]), iou, epoch)
        
        is_best = mIoU_without_bg > previous_best
        previous_best = max(mIoU_without_bg, previous_best)
        if rank == 0:
            checkpoint = {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
                'previous_best': previous_best,
            }
            torch.save(checkpoint, os.path.join(args.save_path, 'latest.pth'))
            if is_best:
                torch.save(checkpoint, os.path.join(args.save_path, 'best.pth'))
        # torch.distributed.barrier()

        # if epoch % 20 == 0:
        #     time.sleep(5)
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

if __name__ == '__main__':
    main()
