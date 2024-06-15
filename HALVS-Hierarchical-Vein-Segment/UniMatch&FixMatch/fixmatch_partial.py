import argparse
import logging
import os
import pprint

import torch
from torch import nn
import torch.backends.cudnn as cudnn
from torch.optim import SGD
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import yaml

from dataset.semi import SemiDataset
from model.semseg.deeplabv3plus import DeepLabV3Plus
from supervised import evaluate
from util.classes import CLASSES
from util.ohem import ProbOhemCrossEntropy2d
from util.utils import count_params, init_log, AverageMeter
from util.dist_helper import setup_distributed,exclusive_loss



parser.add_argument('--config', type=str, required=True)
parser.add_argument('--labeled-id-path', type=str, required=True)
parser.add_argument('--partial_labeled_id_path',type=str, required=True)
parser.add_argument('--unlabeled-id-path', type=str, required=True)
parser.add_argument('--save-path', type=str, required=True)
parser.add_argument('--local_rank', default=0, type=int)
parser.add_argument('--port', default=None, type=int)


def main():
    args = parser.parse_args()

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
    if rank == 0:
        logger.info('Total params: {:.1f}M\n'.format(count_params(model)))

    optimizer = SGD([{'params': model.backbone.parameters(), 'lr': cfg['lr']},
                     {'params': [param for name, param in model.named_parameters() if 'backbone' not in name],
                      'lr': cfg['lr'] * cfg['lr_multi']}], lr=cfg['lr'], momentum=0.9, weight_decay=1e-4)

    local_rank = int(os.environ["LOCAL_RANK"])
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model.cuda()

    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], broadcast_buffers=False,
                                                      output_device=local_rank, find_unused_parameters=False)

    if cfg['criterion']['name'] == 'CELoss':
        criterion_l = nn.CrossEntropyLoss(**cfg['criterion']['kwargs']).cuda(local_rank)
    elif cfg['criterion']['name'] == 'OHEM':
        criterion_l = ProbOhemCrossEntropy2d(**cfg['criterion']['kwargs']).cuda(local_rank)
    else:
        raise NotImplementedError('%s criterion is not implemented' % cfg['criterion']['name'])

    criterion_u = nn.CrossEntropyLoss(reduction='none').cuda(local_rank)

    trainset_u = SemiDataset(cfg['dataset'], cfg['data_root'], 'train_u',
                             cfg['crop_size'], args.unlabeled_id_path)
    trainset_l = SemiDataset(cfg['dataset'], cfg['data_root'], 'train_l',
                             cfg['crop_size'], args.labeled_id_path, nsample=len(trainset_u.ids))
    trainset_p = SemiDataset(cfg['dataset'], cfg['data_root'], 'train_p',
                             cfg['crop_size'], args.partial_labeled_id_path, nsample=len(trainset_u.ids))
    
    valset = SemiDataset(cfg['dataset'], cfg['data_root'], 'val')

    trainsampler_l = torch.utils.data.distributed.DistributedSampler(trainset_l)
    trainloader_l = DataLoader(trainset_l, batch_size=cfg['batch_size'],
                               pin_memory=True, num_workers=1, drop_last=True, sampler=trainsampler_l)
    trainsampler_u = torch.utils.data.distributed.DistributedSampler(trainset_u)
    trainloader_u = DataLoader(trainset_u, batch_size=cfg['batch_size'],
                               pin_memory=True, num_workers=1, drop_last=True, sampler=trainsampler_u)
    trainsampler_p = torch.utils.data.distributed.DistributedSampler(trainset_p)
    trainloader_p = DataLoader(trainset_p, batch_size=cfg['batch_size'],
                               pin_memory=True, num_workers=1, drop_last=True, sampler=trainsampler_p)

    valsampler = torch.utils.data.distributed.DistributedSampler(valset)
    valloader = DataLoader(valset, batch_size=1, pin_memory=True, num_workers=1,
                           drop_last=False, sampler=valsampler)

    total_iters = len(trainloader_u) * cfg['epochs']
    previous_best = 0.0
    past_max_iou = 0.0
    epoch = -1

    if os.path.exists(os.path.join(args.save_path, 'latest.pth')):
        checkpoint = torch.load(os.path.join(args.save_path, 'latest.pth'))
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        epoch = checkpoint['epoch']
        previous_best = checkpoint['previous_best']
        
        if rank == 0:
            logger.info('************ Load from checkpoint at epoch %i\n' % epoch)
    
    for epoch in range(epoch + 1, cfg['epochs']):
        if rank == 0:
            logger.info('===========> Epoch: {:}, LR: {:.5f}, Previous best: {:.2f}'.format(
                epoch, optimizer.param_groups[0]['lr'], previous_best))

        total_loss  = AverageMeter()

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

        loader = zip(trainloader_l,trainloader_p, trainloader_u, trainloader_u)


        for i, ((img_x, mask_x),
                (img_p_w,mask_p,img_p_s),
                (img_u_w, img_u_s, _, ignore_mask, cutmix_box, _),
                (img_u_w_mix, img_u_s_mix, _, ignore_mask_mix, _, _)) in enumerate(loader):

            img_x, mask_x = img_x.cuda(), mask_x.cuda()
            img_p_w,mask_p,img_p_s = img_p_w.cuda(),mask_p.cuda(),img_p_s.cuda()
            img_u_w, img_u_s = img_u_w.cuda(), img_u_s.cuda()
            ignore_mask, cutmix_box = ignore_mask.cuda(), cutmix_box.cuda()
            img_u_w_mix, img_u_s_mix = img_u_w_mix.cuda(), img_u_s_mix.cuda()
            ignore_mask_mix = ignore_mask_mix.cuda()

            with torch.no_grad():
                model.eval()

                pred_u_w_mix = model(img_u_w_mix).detach()
                conf_u_w_mix = pred_u_w_mix.softmax(dim=1).max(dim=1)[0]
                mask_u_w_mix = pred_u_w_mix.argmax(dim=1)

                pred_p_w = model(img_p_w)
                conf_p_w = pred_p_w.softmax(dim=1).max(dim=1)[0]#(B,H,W)
                label_p_w = pred_p_w.argmax(dim=1)

            img_u_s[cutmix_box.unsqueeze(1).expand(img_u_s.shape) == 1] = \
                img_u_s_mix[cutmix_box.unsqueeze(1).expand(img_u_s.shape) == 1]

            model.train()

            pred_p_s = model(img_p_s)

            num_lb, num_ulb = img_x.shape[0], img_u_w.shape[0]

            pred_x, pred_u_w = model(torch.cat((img_x, img_u_w))).split([num_lb, num_ulb])
            pred_u_s = model(img_u_s)

            

            pred_u_w = pred_u_w.detach()
            conf_u_w = pred_u_w.softmax(dim=1).max(dim=1)[0]
            mask_u_w = pred_u_w.argmax(dim=1)

            mask_u_w_cutmixed, conf_u_w_cutmixed, ignore_mask_cutmixed = \
                mask_u_w.clone(), conf_u_w.clone(), ignore_mask.clone()

            mask_u_w_cutmixed[cutmix_box == 1] = mask_u_w_mix[cutmix_box == 1]
            conf_u_w_cutmixed[cutmix_box == 1] = conf_u_w_mix[cutmix_box == 1]
            ignore_mask_cutmixed[cutmix_box == 1] = ignore_mask_mix[cutmix_box == 1]

            loss_x = criterion_l(pred_x, mask_x)

            loss_u_s = criterion_u(pred_u_s, mask_u_w_cutmixed)
            loss_u_s = loss_u_s * ((conf_u_w_cutmixed >= cfg['conf_thresh']) & (ignore_mask_cutmixed != 255))
            loss_u_s = loss_u_s.sum() / (ignore_mask_cutmixed != 255).sum().item()

            loss_u_p = loss_x + loss_u_s 




            mask_p_label = (mask_p == 1) | (mask_p == 2)
            mask_p_unlabel = ~mask_p_label
            mask_p_unlabel_high_conf = mask_p_unlabel & ((label_p_w == 0) | (label_p_w == 3)) &\
                                    (conf_p_w >= cfg["conf_threshold"])
            mask_p_unlabel_others = mask_p_unlabel ^ mask_p_unlabel_high_conf
            # criterion for supervised loss
            criterion_partial = torch.nn.CrossEntropyLoss(reduction='none').cuda(local_rank)
                           
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

            total_loss.update(loss.item())

            total_loss_u_p.update(loss_u_p.item())
            total_loss_x.update(loss_x.item())
            total_loss_s.update(loss_u_s.item())
            

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
                writer.add_scalar('train/loss_s', loss_u_s.item(), iters)
                writer.add_scalar('train/mask_ratio', mask_ratio, iters)

            if (i % (len(trainloader_u) // 8) == 0) and (rank == 0):
                logger.info('Iters: {:}, Total loss: {:.3f},Partial loss: {:.3f},Partial sup_loss: {:.3f} ,Partial unsup_loss_1: {:.3f},Partial unsup_loss_2: {:.3f} ,Loss x: {:.3f}, Loss s: {:.3f}, Mask ratio: '
                            '{:.3f}'.format(i, total_loss.avg, total_loss_p.avg,total_loss_p_sup_loss.avg,total_loss_p_unsup_loss_1.avg, total_loss_p_unsup_loss_2.avg,total_loss_x.avg, 
                                            total_loss_s.avg, total_mask_ratio.avg))
        
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
            for i, iou in enumerate(iou_class):
                writer.add_scalar('eval/%s_IoU' % (CLASSES[cfg['dataset']][i]), iou, epoch)

        is_best = mIoU > previous_best
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


if __name__ == '__main__':
    main()
