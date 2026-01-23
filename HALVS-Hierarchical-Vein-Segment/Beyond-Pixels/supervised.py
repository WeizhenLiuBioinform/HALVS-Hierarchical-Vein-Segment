import argparse
import logging
import os
import pprint

import torch
import numpy as np
from torch import nn
import torch.distributed as dist
import torch.backends.cudnn as cudnn
from torch.optim import SGD
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import yaml

from dataset.semi import SemiDataset
from model.semseg.deeplabv3plus import DeepLabV3Plus
from util.classes import CLASSES
from util.ohem import ProbOhemCrossEntropy2d
from util.utils import count_params, AverageMeter, intersectionAndUnion, init_log
from util.dist_helper import setup_distributed
import matplotlib.pyplot as plt
import cv2
import torchvision.transforms.functional as TF
from util.utils import color_map
from PIL import Image
parser = argparse.ArgumentParser(description='Revisiting Weak-to-Strong Consistency in Semi-Supervised Semantic Segmentation')
parser.add_argument('--config', type=str, required=True)
parser.add_argument('--labeled-id-path', type=str, required=True)
parser.add_argument('--unlabeled-id-path', type=str, default=None)
parser.add_argument('--save-path', type=str, required=True)
parser.add_argument('--local_rank', default=0, type=int)
parser.add_argument('--port', default=None, type=int)

def seg_to_rgb(seg, colors):
    seg=seg.cpu().detach().numpy()#(1, 500, 375)
    im = np.uint8(np.zeros((seg.shape[0], seg.shape[1], seg.shape[2], 3)))#(1, 500, 375, 3)
    cls = np.unique(seg)
    for cl in cls:
        color = colors[int(cl)]
        if len(color.shape) > 1:
            color = color[0]
        im[seg == cl] = color
    return im

######################code to get the segmentation numpies
def get_pseudolabels(model, loader, mode, cfg,PATH_SAVE):
    model.eval()
    assert mode in ['original', 'center_crop', 'sliding_window']
    intersection_meter = AverageMeter()
    union_meter = AverageMeter()
    CLASSES= ['background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 
                          'car', 'cat', 'chair', 'cow', 'dining table', 'dog', 'horse', 'motorbike', 
                          'person', 'potted plant', 'sheep', 'sofa', 'train', 'tv/monitor']
    colors = color_map('pascal')
    
    COLOR_FIN ={}
    _i_=0
    for _class_ in CLASSES:
        COLOR_FIN[_i_]= colors[_i_,:]
        _i_+=1
    COLOR_FIN[255]=np.array([0,0,0])
    with torch.no_grad():
        for img, mask, id in loader:
            
            img = img.cuda()

            if mode == 'sliding_window':
                grid = cfg['crop_size']
                b, _, h, w = img.shape
                final = torch.zeros(b, 19, h, w).cuda()
                row = 0
                while row < h:
                    col = 0
                    while col < w:
                        pred = model(img[:, :, row: min(h, row + grid), col: min(w, col + grid)])
                        final[:, :, row: min(h, row + grid), col: min(w, col + grid)] += pred.softmax(dim=1)
                        col += int(grid * 2 / 3)
                    row += int(grid * 2 / 3)

                pred = final.argmax(dim=1)

            else:
                if mode == 'center_crop':
                    h, w = img.shape[-2:]
                    start_h, start_w = (h - cfg['crop_size']) // 2, (w - cfg['crop_size']) // 2
                    img = img[:, :, start_h:start_h + cfg['crop_size'], start_w:start_w + cfg['crop_size']]
                    mask = mask[:, start_h:start_h + cfg['crop_size'], start_w:start_w + cfg['crop_size']]

                pred = model(img).argmax(dim=1)#[1, 500, 375]
                conf= model(img).softmax(dim=1).max(dim=1)[0].cpu()#[1, 500, 375]
                #print(pred.shape,img.shape)#[1, 500, 375]) [1, 3, 500, 375]
                #print(id)#'JPEGImages/2011_001988.jpg SegmentationClass/2011_001988.png'
                img = Image.open(os.path.join(cfg['data_root'], id[0].split(' ')[0])).convert('RGB')#(500, 333, 3)#(333, 500)-----PIL
                mask = np.array(Image.open(os.path.join(cfg['data_root'], id[0].split(' ')[1])))#(500,375)
                mask = torch.from_numpy(mask[np.newaxis,:,:])#(1, 500, 375)
                heatmap = torch.cat((conf, torch.zeros(2, conf.shape[1], conf.shape[2])))
                h_img = TF.to_pil_image(heatmap)
                res = Image.blend(img, h_img, 0.3)
                res.save(PATH_SAVE+'/'+id[0].split('/')[-1].split('.')[0]+'_heat.png')
                pred_img=seg_to_rgb(pred, COLOR_FIN)
                pred_img = np.squeeze(pred_img)
                pred_img = Image.fromarray(pred_img)
                pred_img.save(PATH_SAVE+'/'+id[0].split('/')[-1].split('.')[0]+'_pred.png')
                GT = seg_to_rgb(mask, COLOR_FIN)
                GT = np.squeeze(GT)
                GT = Image.fromarray(GT)
                GT.save(PATH_SAVE+'/'+id[0].split('/')[-1].split('.')[0]+'_gt.png')
                pred[conf<cfg['conf_thresh']]=255
                pred_img=seg_to_rgb(pred, COLOR_FIN)
                pred_img = np.squeeze(pred_img)
                pred_img = Image.fromarray(pred_img)
                pred_img.save(PATH_SAVE+'/'+id[0].split('/')[-1].split('.')[0]+'_predafterconf.png') 


######################code to get the segmentation numpies
def get_segmentation(model, loader, mode, cfg,PATH_SAVE):
    model.eval()
    assert mode in ['original', 'center_crop', 'sliding_window']
    intersection_meter = AverageMeter()
    union_meter = AverageMeter()

    with torch.no_grad():
        for img, mask, id in loader:
            
            img = img.cuda()

            if mode == 'sliding_window':
                grid = cfg['crop_size']
                b, _, h, w = img.shape
                final = torch.zeros(b, 19, h, w).cuda()
                row = 0
                while row < h:
                    col = 0
                    while col < w:
                        pred = model(img[:, :, row: min(h, row + grid), col: min(w, col + grid)])
                        final[:, :, row: min(h, row + grid), col: min(w, col + grid)] += pred.softmax(dim=1)
                        col += int(grid * 2 / 3)
                    row += int(grid * 2 / 3)

                pred = final.argmax(dim=1)

            else:
                if mode == 'center_crop':
                    h, w = img.shape[-2:]
                    start_h, start_w = (h - cfg['crop_size']) // 2, (w - cfg['crop_size']) // 2
                    img = img[:, :, start_h:start_h + cfg['crop_size'], start_w:start_w + cfg['crop_size']]
                    mask = mask[:, start_h:start_h + cfg['crop_size'], start_w:start_w + cfg['crop_size']]

                pred = model(img).argmax(dim=1)
            pred=pred.cpu().numpy()
            np.save(PATH_SAVE+'/'+id[0].split('/')[-1].split('.')[0], pred)
            
######################code to get the segmentation bbox
def get_segmentation_bbox(model, loader, mode, cfg,PATH_SAVE, temp_map):
    model.eval()
    assert mode in ['original', 'center_crop', 'sliding_window']
    intersection_meter = AverageMeter()
    union_meter = AverageMeter()
    CLASSES= ['background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 
                          'car', 'cat', 'chair', 'cow', 'dining table', 'dog', 'horse', 'motorbike', 
                          'person', 'potted plant', 'sheep', 'sofa', 'train', 'tv/monitor']
    colors = color_map('pascal')

    
    COLOR_FIN ={}
    _i_=0
    for _class_ in CLASSES:
        COLOR_FIN[_i_]= colors[_i_,:]
        _i_+=1
    COLOR_FIN[255]=np.array([0,0,0])
    all_correct_area_ratio={}
    all_ratio={}
    with torch.no_grad():
        for img, mask in loader:
            img = img.cuda()
            mask = mask.cuda()
            mask_patches = torch.stack([mask==i for i in range(cfg['nclass'])], dim=1)#1,19,801,801
            mask_patches = mask_patches.float().cuda()#[1, 21, 321, 321]
            if mode == 'sliding_window':
                grid = cfg['crop_size']
                b, _, h, w = img.shape
                final = torch.zeros(b, 19, h, w).cuda()
                row = 0
                while row < h:
                    col = 0
                    while col < w:
                        pred = model(img[:, :, row: min(h, row + grid), col: min(w, col + grid)])
                        final[:, :, row: min(h, row + grid), col: min(w, col + grid)] += pred.softmax(dim=1)
                        col += int(grid * 2 / 3)
                    row += int(grid * 2 / 3)

                pred = final.argmax(dim=1)

            else:
                if mode == 'center_crop':
                    h, w = img.shape[-2:]
                    start_h, start_w = (h - cfg['crop_size']) // 2, (w - cfg['crop_size']) // 2
                    img = img[:, :, start_h:start_h + cfg['crop_size'], start_w:start_w + cfg['crop_size']]
                    mask = mask[:, start_h:start_h + cfg['crop_size'], start_w:start_w + cfg['crop_size']]
                    
                classifier_feats,out = model(img, classify=True)#[1, 6561, 21],[1, 21, 321, 321]
                pred = out.argmax(dim=1)
                conf= out.softmax(dim=1).max(dim=1)[0].cpu()#[1, 500, 375]
                classifier_feats = torch.sigmoid(torch.squeeze(classifier_feats))
                for _i_ in range(classifier_feats.shape[0]):
                    u_x,v_x,u_y,v_y = int(temp_map[_i_,0].item()),int(temp_map[_i_,1].item()),int(temp_map[_i_,2].item()),int(temp_map[_i_,3].item())
                    act_size = (v_x-u_x)*(v_y-u_y)
                    for _j_ in range(cfg['nclass']):
                        temp_mask = mask_patches[:,_j_, u_x:v_x, u_y:v_y].sum().item()
                        classifier_pred= classifier_feats[_i_,_j_].item()
                        if(classifier_pred>=0.5):
                            classifier_pred=1
                        else:
                            classifier_pred=0
                        ratio = int(temp_mask/act_size)
                        if(ratio not in all_ratio.keys()):
                            all_ratio[ratio]={}
                            all_correct_area_ratio[ratio]={}
                        if(_j_ not in all_ratio[ratio].keys()):
                            all_ratio[ratio][_j_]=0
                            all_correct_area_ratio[ratio][_j_]=0
                        all_ratio[ratio][_j_]+=1
                        truth=False
                        if(classifier_pred==1):
                            if(temp_mask>0):
                                truth = True
                        if(truth):
                            all_correct_area_ratio[ratio][_j_]+=1
#     for _ratio_ in all_correct_area_ratio.keys():
#         for _key_ in all_correct_area_ratio[_ratio_].keys():
#             all_correct_area_ratio[_ratio_][_key_]= round((all_correct_area_ratio[_ratio_][_key_]/all_ratio[_ratio_][_key_])*100,2)
    with open(PATH_SAVE+'/'+'all_correct_area_ratio.pickle', 'wb') as handle:
        pickle.dump(all_correct_area_ratio, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open(PATH_SAVE+'/'+'all_ratio.pickle', 'wb') as handle:
        pickle.dump(all_ratio, handle, protocol=pickle.HIGHEST_PROTOCOL) 
            
def evaluate(model, loader, mode, cfg):
    model.eval()
    assert mode in ['original', 'center_crop', 'sliding_window']
    intersection_meter = AverageMeter()
    union_meter = AverageMeter()

    with torch.no_grad():
        for img, mask, id in loader:
            
            img = img.cuda()

            if mode == 'sliding_window':
                grid = cfg['crop_size']
                b, _, h, w = img.shape
                final = torch.zeros(b, 19, h, w).cuda()
                row = 0
                while row < h:
                    col = 0
                    while col < w:
                        pred = model(img[:, :, row: min(h, row + grid), col: min(w, col + grid)])
                        final[:, :, row: min(h, row + grid), col: min(w, col + grid)] += pred.softmax(dim=1)
                        col += int(grid * 2 / 3)
                    row += int(grid * 2 / 3)

                pred = final.argmax(dim=1)

            else:
                if mode == 'center_crop':
                    h, w = img.shape[-2:]
                    start_h, start_w = (h - cfg['crop_size']) // 2, (w - cfg['crop_size']) // 2
                    img = img[:, :, start_h:start_h + cfg['crop_size'], start_w:start_w + cfg['crop_size']]
                    mask = mask[:, start_h:start_h + cfg['crop_size'], start_w:start_w + cfg['crop_size']]

                pred = model(img).argmax(dim=1)

            intersection, union, target = \
                intersectionAndUnion(pred.cpu().numpy(), mask.numpy(), cfg['nclass'], 255)

            reduced_intersection = torch.from_numpy(intersection).cuda()
            reduced_union = torch.from_numpy(union).cuda()
            reduced_target = torch.from_numpy(target).cuda()

            dist.all_reduce(reduced_intersection)
            dist.all_reduce(reduced_union)
            dist.all_reduce(reduced_target)

            intersection_meter.update(reduced_intersection.cpu().numpy())
            union_meter.update(reduced_union.cpu().numpy())

    iou_class = intersection_meter.sum / (union_meter.sum + 1e-10) * 100.0
    mIOU = np.mean(iou_class)

    return mIOU, iou_class


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
    model.cuda(local_rank)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], broadcast_buffers=False,
                                                      output_device=local_rank, find_unused_parameters=False)

    if cfg['criterion']['name'] == 'CELoss':
        criterion = nn.CrossEntropyLoss(**cfg['criterion']['kwargs']).cuda(local_rank)
    elif cfg['criterion']['name'] == 'OHEM':
        criterion = ProbOhemCrossEntropy2d(**cfg['criterion']['kwargs']).cuda(local_rank)
    else:
        raise NotImplementedError('%s criterion is not implemented' % cfg['criterion']['name'])

    trainset = SemiDataset(cfg['dataset'], cfg['data_root'], 'train_l', cfg['crop_size'], args.labeled_id_path)
    valset = SemiDataset(cfg['dataset'], cfg['data_root'], 'val')

    trainsampler = torch.utils.data.distributed.DistributedSampler(trainset)
    trainloader = DataLoader(trainset, batch_size=cfg['batch_size'],
                             pin_memory=True, num_workers=1, drop_last=True, sampler=trainsampler)
    valsampler = torch.utils.data.distributed.DistributedSampler(valset)
    valloader = DataLoader(valset, batch_size=1, pin_memory=True, num_workers=1,
                           drop_last=False, sampler=valsampler)

    iters = 0
    total_iters = len(trainloader) * cfg['epochs']
    previous_best = 0.0
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

        model.train()
        total_loss = AverageMeter()

        trainsampler.set_epoch(epoch)

        for i, (img, mask) in enumerate(trainloader):

            img, mask = img.cuda(), mask.cuda()

            pred = model(img)

            loss = criterion(pred, mask)
            
            torch.distributed.barrier()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss.update(loss.item())

            iters = epoch * len(trainloader) + i
            lr = cfg['lr'] * (1 - iters / total_iters) ** 0.9
            optimizer.param_groups[0]["lr"] = lr
            optimizer.param_groups[1]["lr"] = lr * cfg['lr_multi']

            if rank == 0:
                writer.add_scalar('train/loss_all', loss.item(), iters)
                writer.add_scalar('train/loss_x', loss.item(), iters)
            
            if (i % (max(2, len(trainloader) // 8)) == 0) and (rank == 0):
                logger.info('Iters: {:}, Total loss: {:.3f}'.format(i, total_loss.avg))

        eval_mode = 'sliding_window' if cfg['dataset'] == 'cityscapes' else 'original'
        mIoU, iou_class = evaluate(model, valloader, eval_mode, cfg)

        if rank == 0:
            for (cls_idx, iou) in enumerate(iou_class):
                logger.info('***** Evaluation ***** >>>> Class [{:} {:}] '
                            'IoU: {:.2f}'.format(cls_idx, CLASSES[cfg['dataset']][cls_idx], iou))
            logger.info('***** Evaluation {} ***** >>>> MeanIoU: {:.2f}\n'.format(eval_mode, mIoU))
            
            writer.add_scalar('eval/mIoU', mIoU, epoch)
            for i, iou in enumerate(iou_class):
                writer.add_scalar('eval/%s_IoU' % (CLASSES[cfg['dataset']][i]), iou, epoch)

        is_best = mIoU > previous_best
        previous_best = max(mIoU, previous_best)
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
