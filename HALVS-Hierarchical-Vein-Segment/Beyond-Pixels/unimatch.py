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
#from model.semseg.linear_model import LinearClassifier
from supervised import evaluate
from util.classes import CLASSES
from util.ohem import ProbOhemCrossEntropy2d
from util.utils import count_params, init_log, AverageMeter
from util.dist_helper import setup_distributed
from PIL import Image as im
import matplotlib.pyplot as plt
import numpy as np
import pickle
import time
import gc
import torch.nn as nn
from model.semseg.maxpool_label import Poolinglabel
from model.semseg.maxpool_label import calculate_receptive_field
import torch.nn.functional as F
from util.new_multilabel import compute_batch_loss
import time

parser = argparse.ArgumentParser(description='Revisiting Weak-to-Strong Consistency in Semi-Supervised Semantic Segmentation')
parser.add_argument('--config', type=str, required=True)
parser.add_argument('--labeled-id-path', type=str, required=True)
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
    #print(cfg)
    #{'dataset': 'cityscapes', 'nclass': 19, 'crop_size': 801, 'data_root': '/nfs/bigtensor/add_disk0/ironman/data/cityscapes', 'epochs': 240, 'batch_size': 2, 'lr': 0.005, 'lr_multi': 1.0, 'criterion': {'name': 'OHEM', 'kwargs': {'ignore_index': 255, 'thresh': 0.7, 'min_kept': 200000}}, 'conf_thresh': 0, 'backbone': 'resnet101', 'replace_stride_with_dilation': [False, False, True], 'dilations': [6, 12, 18]}


    model = DeepLabV3Plus(cfg)
    print("MODEL")
    optimizer = SGD([{'params': model.backbone.parameters(), 'lr': cfg['lr']},
                    {'params': [param for name, param in model.named_parameters() if 'backbone' not in name],
                    'lr': cfg['lr'] * cfg['lr_multi']}], lr=cfg['lr'], momentum=0.9, weight_decay=1e-4) 
#     model_classifier = LinearClassifier(input_dim=256, output_dim=19)
#     print("MODEL classifier")

#     optimizer = SGD([{'params': model.backbone.parameters(), 'lr': cfg['lr']},
#                      {'params': model_classifier.parameters(), 'lr': cfg['lr']},
#                     {'params': [param for name, param in model.named_parameters() if 'backbone' not in name],
#                     'lr': cfg['lr'] * cfg['lr_multi']}], lr=cfg['lr'], momentum=0.9, weight_decay=1e-4)
  
    if rank == 0:
        logger.info('Total params: {:.1f}M\n'.format(count_params(model)))

    local_rank = int(os.environ["LOCAL_RANK"])
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model.cuda()
#     batchnorm_conv1 = nn.Sequential(
#             nn.MaxPool2d( kernel_size=3, stride=2, padding=1),
#             nn.MaxPool2d( kernel_size=3, stride=1, padding=1),
#             nn.MaxPool2d( kernel_size=3, stride=1, padding=1),
#             )
#     batchnorm_maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
#     batchnorm_layer_1 = nn.Sequential(
#             nn.MaxPool2d( kernel_size=3, stride=1, padding=1),
#             nn.MaxPool2d( kernel_size=3, stride=1, padding=1),
#             nn.MaxPool2d( kernel_size=3, stride=1, padding=1),
#             )
    poolinglabel_model = Poolinglabel(19)
    poolinglabel_model.cuda()

#     poolinglabel_model = torch.nn.parallel.DistributedDataParallel(poolinglabel_model, device_ids=[local_rank], broadcast_buffers=False,
#                                                      output_device=local_rank, find_unused_parameters=False)
#     model_classifier.cuda()
    print("START 1")
    

    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], broadcast_buffers=False,
                                                      output_device=local_rank, find_unused_parameters=False)
#     #    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], broadcast_buffers=False,
#                                                      output_device=local_rank, find_unused_parameters=True)
#     model_classifier = torch.nn.parallel.DistributedDataParallel(model_classifier, device_ids=[local_rank], broadcast_buffers=False,
#                                                       output_device=local_rank, find_unused_parameters=False)
    print("START 2")
    #poolinglabel_model.cuda(local_rank)
    if cfg['criterion']['name'] == 'CELoss':
        criterion_l = nn.CrossEntropyLoss(**cfg['criterion']['kwargs']).cuda(local_rank)
    elif cfg['criterion']['name'] == 'OHEM':
        criterion_l = ProbOhemCrossEntropy2d(**cfg['criterion']['kwargs']).cuda(local_rank)
    else:
        raise NotImplementedError('%s criterion is not implemented' % cfg['criterion']['name'])

    criterion_u = nn.CrossEntropyLoss(reduction='none').cuda(local_rank)
    criterion_multilabel=nn.BCEWithLogitsLoss(reduction='none').cuda(local_rank)
    
    trainset_u = SemiDataset(cfg['dataset'], cfg['data_root'], 'train_u',
                             cfg['crop_size'], args.unlabeled_id_path)
    
    trainset_l = SemiDataset(cfg['dataset'], cfg['data_root'], 'train_l',
                             cfg['crop_size'], args.labeled_id_path, nsample=len(trainset_u.ids))    

 
    valset = SemiDataset(cfg['dataset'], cfg['data_root'], 'val')

    trainsampler_l = torch.utils.data.distributed.DistributedSampler(trainset_l)
    trainloader_l = DataLoader(trainset_l, batch_size=cfg['batch_size'],
                               pin_memory=True, num_workers=1, drop_last=True, sampler=trainsampler_l)
    trainsampler_u = torch.utils.data.distributed.DistributedSampler(trainset_u)
    trainloader_u = DataLoader(trainset_u, batch_size=cfg['batch_size'],
                               pin_memory=True, num_workers=1, drop_last=True, sampler=trainsampler_u)
    valsampler = torch.utils.data.distributed.DistributedSampler(valset)
    valloader = DataLoader(valset, batch_size=1, pin_memory=True, num_workers=1,
                           drop_last=False, sampler=valsampler)

    total_iters = len(trainloader_u) * cfg['epochs']

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



            

    #calcualate the size of the original image
    t_img, t_mask = trainset_l[0]#[3, 801, 801],[801, 801]
    org_u_left=0
    org_v_right= t_mask.shape[1]-1
    pred_u_left =0
    pred_v_right=200
    list_receptive_field=[]
    for st in range(pred_v_right+1):
        list_receptive_field.append([calculate_receptive_field(st,st)])
    receptive_pad_left = org_u_left-list_receptive_field[0][0][0]
    receptive_pad_right = list_receptive_field[-1][0][1]-org_v_right
    receptive_kernel_size = list_receptive_field[0][0][1]-list_receptive_field[0][0][0]
    receptive_stride = list_receptive_field[1][0][0]-list_receptive_field[0][0][0]
    weigth_avgpool = nn.AvgPool2d(kernel_size=receptive_kernel_size, stride=receptive_stride, padding=receptive_pad_left, count_include_pad=False)
    temp_map = torch.zeros(4,201,201)
    for _i_ in range(201):
        for _j_ in range(201):
            _u_x,_v_x = calculate_receptive_field(_i_,_i_)
            if(_u_x<0):
                _u_x=0
            if(_u_x>=801):
                _u_x=800
            if(_v_x<0):
                _v_x=0
            if(_v_x>=801):
                _v_x=800            
            temp_map[0,_i_,_j_]=_u_x
            temp_map[1,_i_,_j_]=_v_x
            _u_y,_v_y = calculate_receptive_field(_j_,_j_)
            if(_u_y<0):
                _u_y=0
            if(_u_y>=801):
                _u_y=800
            if(_v_y<0):
                _v_y=0
            if(_v_y>=801):
                _v_y=800            
            temp_map[2,_i_,_j_]=_u_y
            temp_map[3,_i_,_j_]=_v_y

    temp_map = temp_map.view(temp_map.shape[0],-1)
    temp_map = temp_map.permute(1,0)#[40401, 4]
    all_class_delta_rel = {}
    for _class_ in range(cfg['nclass']):
        all_class_delta_rel[_class_]=0.00001
    all_class_labeled_unconf={}
    for _class_ in range(cfg['nclass']):
        all_class_labeled_unconf[_class_]=AverageMeter(length=4)
    all_class_max_labeled_unconf = {}
    all_class_max_labeled_unconf_epoch = {}
    for _class_ in range(cfg['nclass']):
        all_class_max_labeled_unconf_epoch[_class_]=0
        all_class_max_labeled_unconf[_class_]=0

    epoch_change=0
    old_epoch = epoch
    for epoch in range(epoch + 1, cfg['epochs']):
#         cfg['clean_rate'] -= delta_rel
#         if(cfg['clean_rate']<=0.1):
#             cfg['clean_rate']=0.1
        
        if rank == 0:
            logger.info('===========> Epoch: {:}, LR: {:.5f}, Previous best: {:.2f}'.format(
                epoch, optimizer.param_groups[0]['lr'], previous_best))

        total_loss = AverageMeter()
        total_loss_x = AverageMeter()
        total_loss_s = AverageMeter()
        total_loss_w_fp = AverageMeter()
        total_mask_ratio = AverageMeter()
        total_loss_multi = AverageMeter()
        total_loss_multi_unlab = AverageMeter()
        total_all_class_labeled_unconf={}
        for _class_ in range(cfg['nclass']):
            total_all_class_labeled_unconf[_class_]=AverageMeter()    
        total_overall_count=0

        trainloader_l.sampler.set_epoch(epoch)
        trainloader_u.sampler.set_epoch(epoch)
        loader = zip(trainloader_l, trainloader_u, trainloader_u)
        for i, ((img_x, mask_x),
                (img_u_w, img_u_s1, img_u_s2, ignore_mask, cutmix_box1, cutmix_box2,gt_mask_u_w),
                (img_u_w_mix, img_u_s1_mix, img_u_s2_mix, ignore_mask_mix, _, _, gt_mask_u_w_mix)) in enumerate(loader):
            total_overall_count+=1
            with torch.no_grad():
                mask_patches = torch.stack([mask_x==i for i in range(19)], dim=1)#1,19,801,801
                mask_patches = mask_patches.float().cuda()
                mask_patches = weigth_avgpool(mask_patches)#[1, 19, 201, 201]
            mask_patches = mask_patches.cuda()
            img_x, mask_x = img_x.cuda(), mask_x.cuda()
            #print(img_x.shape, mask_x.shape)#torch.Size([2, 3, 801, 801]) torch.Size([2, 801, 801])
            img_u_w = img_u_w.cuda()
            img_u_s1, img_u_s2, ignore_mask = img_u_s1.cuda(), img_u_s2.cuda(), ignore_mask.cuda()
            cutmix_box1, cutmix_box2 = cutmix_box1.cuda(), cutmix_box2.cuda()
            img_u_w_mix = img_u_w_mix.cuda()
            img_u_s1_mix, img_u_s2_mix = img_u_s1_mix.cuda(), img_u_s2_mix.cuda()
            ignore_mask_mix = ignore_mask_mix.cuda()
            gt_mask_u_w = gt_mask_u_w.cuda()#1,801,801
            gt_mask_u_w_mix = gt_mask_u_w_mix.cuda()#1,801,801
            mask_u_s1 = gt_mask_u_w.clone()
            mask_u_s2 = gt_mask_u_w.clone()
            
            with torch.no_grad():
                model.eval()
                pred_u_w_mix = model(img_u_w_mix).detach()
                conf_u_w_mix = pred_u_w_mix.softmax(dim=1).max(dim=1)[0]
                mask_u_w_mix = pred_u_w_mix.argmax(dim=1)
            #applying cutmix operation
            img_u_s1[cutmix_box1.unsqueeze(1).expand(img_u_s1.shape) == 1] = \
                img_u_s1_mix[cutmix_box1.unsqueeze(1).expand(img_u_s1.shape) == 1]
            img_u_s2[cutmix_box2.unsqueeze(1).expand(img_u_s2.shape) == 1] = \
                img_u_s2_mix[cutmix_box2.unsqueeze(1).expand(img_u_s2.shape) == 1]
            mask_u_s1[cutmix_box1==1]= gt_mask_u_w_mix[cutmix_box1==1]
            mask_u_s2[cutmix_box2==1]= gt_mask_u_w_mix[cutmix_box2==1]
            mlabel_img_s1= img_u_w.clone()
            mlabel_img_s2= img_u_w.clone()
            mlabel_img_s1[cutmix_box1.unsqueeze(1).expand(mlabel_img_s1.shape) == 1] = \
                img_u_w_mix[cutmix_box1.unsqueeze(1).expand(img_u_w_mix.shape) == 1]
            mlabel_img_s2[cutmix_box2.unsqueeze(1).expand(mlabel_img_s2.shape) == 1] = \
                img_u_w_mix[cutmix_box2.unsqueeze(1).expand(img_u_w_mix.shape) == 1]
            with torch.no_grad():
                model.eval()
                cfeats_mlabel,out_mlabel = model(torch.cat((mlabel_img_s1, mlabel_img_s2)),classify=True)#[2, 40401, 19]) [2, 19, 801, 801]
                mlabel_pred_u_s1, mlabel_pred_u_s2 = out_mlabel.chunk(2)#[1, 19, 801, 801]) ([1, 19, 801, 801]
                mlabel_cfeats_s1,mlabel_cfeats_s2 = cfeats_mlabel.chunk(2)#[1, 40401, 19]) ([1, 40401, 19]

                
#             img_u_w_mlabel1 = img_u_w.clone()
#             img_u_w_mlabel[cutmix_box1.unsqueeze(1).expand(img_u_s1.shape) == 1] = \
#                 img_u_s1_mix[cutmix_box1.unsqueeze(1).expand(img_u_s1.shape) == 1
            with torch.no_grad():#gtmultilabels
                mlabels_u_s = poolinglabel_model(torch.cat((mask_u_s1, mask_u_s2)))#[2, 19, 201, 201]
                mlabels_u_s = mlabels_u_s.cuda().detach()
                mlabels_u_s = mlabels_u_s.view(mlabels_u_s.shape[0],mlabels_u_s.shape[1],-1)
                mlabels_u_s = mlabels_u_s.permute(0,2,1)#torch.Size([2, 40401, 19])
                mlabels_u_s1, mlabels_u_s2 = mlabels_u_s.chunk(2)#[1, 40401, 19] [1, 40401, 19]
            model.train()

            num_lb, num_ulb = img_x.shape[0], img_u_w.shape[0]
            #print("1", img_x.shape, img_u_w.shape)#torch.Size([1, 3, 801, 801]) torch.Size([1, 3, 801, 801])
            #pred_x = model(img_x, classify=True)
            c1ss,c1s_fp,preds, preds_fp = model(torch.cat((img_x, img_u_w)), need_fp=True, classify=True, nlabel=num_lb)

            #print("2",c1s.shape, preds.shape, preds_fp.shape)#torch.Size([2, 40401, 19]) torch.Size([4, 19, 801, 801]) torch.Size([4, 19, 801, 801])
            pred_x, pred_u_w = preds.split([num_lb, num_ulb])#[1, 19, 801, 801]) ([1, 19, 801, 801]
            c1s_x, c1s_u_w = c1ss.split([num_lb, num_ulb])#[1, 40401, 19]) [1, 40401, 19]
            c1s_x_prob = torch.sigmoid(c1s_x)
            c1s_x_prob = c1s_x_prob.detach()
            
            
            pred_u_w_fp = preds_fp[num_lb:] ###########!!!!check
            c1s_u_w_fp = c1s_fp[num_lb:]#[1, 40401, 19]
            with torch.no_grad():
                all_multilabel = poolinglabel_model(mask_x)#torch.Size([1, 19, 201, 201])
                all_multilabel = all_multilabel.cuda().detach()
                all_multilabel = all_multilabel.view(all_multilabel.shape[0],all_multilabel.shape[1],-1)
                all_multilabel = all_multilabel.permute(0,2,1)#torch.Size([2, 40401, 19])
                #all_class_labeled_unconf[_class_].update(avg_TP_conf[_class_].item())
#                 if(avg_TP_conf[_class_].item() > all_class_max_labeled_unconf[_class_]):
#                     all_class_max_labeled_unconf[_class_]=avg_TP_conf[_class_].item()
            mask_patches = mask_patches.view(mask_patches.shape[0], mask_patches.shape[1],-1)
            mask_patches = mask_patches.permute(0,2,1)#[1, 40401, 19]
            mask_patches = F.softmax(mask_patches, dim=2)
            
            loss_multi_x = criterion_multilabel(c1s_x, all_multilabel)#[1, 40401, 19]
            loss_multi_x = loss_multi_x * mask_patches#[1, 40401, 19]
            loss_multi_x = torch.sum(loss_multi_x)/torch.sum(mask_patches)
            if torch.isnan(loss_multi_x).any():
                print("LOSS NAN in loss_multi_0 ", torch.unique(c1s_x))
                print("LOSS NAN in loss_multi_1 ",torch.unique(all_multilabel))
                loss_multi_x= torch.nan_to_num(loss_multi_x)
            loss_x = criterion_l(pred_x, mask_x)
            if torch.isnan(loss_x).any():
                print("NAN in loss_x")
                print(pred_x.shape, mask_x.shape)
                print(torch.unique(pred_x), torch.unique(mask_x))
                loss_x = torch.nan_to_num(loss_x)
            if(epoch==0):     
                loss = (loss_x+loss_multi_x ) / 2.0            
                torch.distributed.barrier()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss.update(loss.item())
                total_loss_x.update(loss_x.item())
                total_loss_multi.update(loss_multi_x.item())
#                 total_loss_s.update(0)
#                 total_loss_w_fp.update(0)
#                 total_mask_ratio.update(0)            
            if(epoch>0):
                classifier_feats,out = model(torch.cat((img_u_s1, img_u_s2)),classify=True)#[2, 40401, 19]) [2, 19, 801, 801]
                pred_u_s1, pred_u_s2 = out.chunk(2)#[1, 19, 801, 801]) ([1, 19, 801, 801]
                classifier_feats_s1,classifier_feats_s2 = classifier_feats.chunk(2)#[1, 40401, 19]) ([1, 40401, 19]
                pred_u_w = pred_u_w.detach()
                conf_u_w = pred_u_w.softmax(dim=1).max(dim=1)[0]
                mask_u_w = pred_u_w.argmax(dim=1)
                mask_u_w_cutmixed1, conf_u_w_cutmixed1, ignore_mask_cutmixed1 = \
                    mask_u_w.clone(), conf_u_w.clone(), ignore_mask.clone()
                mask_u_w_cutmixed2, conf_u_w_cutmixed2, ignore_mask_cutmixed2 = \
                    mask_u_w.clone(), conf_u_w.clone(), ignore_mask.clone()
                mask_u_w_cutmixed1[cutmix_box1 == 1] = mask_u_w_mix[cutmix_box1 == 1]#[1, 801, 801]
                conf_u_w_cutmixed1[cutmix_box1 == 1] = conf_u_w_mix[cutmix_box1 == 1]#[1, 801, 801]
                ignore_mask_cutmixed1[cutmix_box1 == 1] = ignore_mask_mix[cutmix_box1 == 1]#[1, 801, 801]
                mask_u_w_cutmixed2[cutmix_box2 == 1] = mask_u_w_mix[cutmix_box2 == 1]
                conf_u_w_cutmixed2[cutmix_box2 == 1] = conf_u_w_mix[cutmix_box2 == 1]
                ignore_mask_cutmixed2[cutmix_box2 == 1] = ignore_mask_mix[cutmix_box2 == 1]
                ########################################################################
                #mask_u_w_cutmixed1[conf_u_w_cutmixed1 < cfg['conf_thresh']]=255
                ignore_mask_cutmixed1[conf_u_w_cutmixed1 < cfg['conf_thresh']]=255
                #mask_u_w_cutmixed2[conf_u_w_cutmixed2 < cfg['conf_thresh']]=255
                ignore_mask_cutmixed2[conf_u_w_cutmixed2 < cfg['conf_thresh']]=255 
                #mask_u_w[conf_u_w < cfg['conf_thresh']]=255
                ignore_mask[conf_u_w < cfg['conf_thresh']]=255
                
                class_u_w_cutmixed1 = torch.zeros(mask_u_w_cutmixed1.shape).cuda()#used for finding the positions with loss high by classifier in s1
                class_u_w_cutmixed2 = torch.zeros(mask_u_w_cutmixed2.shape).cuda()#used for finding the positions with loss high by classifier in s2
                class_u_w_fp = torch.zeros(mask_u_w.shape).cuda()#used for finding the positions with loss high by classifier in w
                with torch.no_grad():
                    temp_mask_u_w_cutmixed1 = mask_u_w_cutmixed1.clone()
                    temp_mask_u_w_cutmixed2 = mask_u_w_cutmixed2.clone()
                    temp_mask_u_w = mask_u_w.clone()
                    temp_mask_u_w_cutmixed1[conf_u_w_cutmixed1 < cfg['conf_thresh']]=255
                    temp_mask_u_w_cutmixed2[conf_u_w_cutmixed2 < cfg['conf_thresh']]=255
                    temp_mask_u_w[conf_u_w < cfg['conf_thresh']]=255
                    mlabels_u_w_s = poolinglabel_model(torch.cat((temp_mask_u_w_cutmixed1, temp_mask_u_w_cutmixed2, temp_mask_u_w)))#[3, 19, 201, 201]
                    mlabels_u_w_s = mlabels_u_w_s.cuda().detach()
                    mlabels_u_w_s = mlabels_u_w_s.view(mlabels_u_w_s.shape[0],mlabels_u_w_s.shape[1],-1)
                    mlabels_u_w_s = mlabels_u_w_s.permute(0,2,1)#torch.Size([3, 40401, 19])
                    mlabels_u_w_cm1, mlabels_u_w_cm2, mlabels_u_w_fp = mlabels_u_w_s.chunk(3)#[1, 40401, 19] [1, 40401, 19] [1, 40401, 19]
                    #compute_batch_loss(torch.squeeze(classifier_feats_s1), torc)
                with torch.no_grad():
                    _, mlabel_correction_idx_s1,_, _ =compute_batch_loss(torch.squeeze(mlabel_cfeats_s1), torch.squeeze(mlabels_u_w_cm1),cfg, all_class_delta_rel, torch.squeeze( mlabels_u_s1))
                    _, mlabel_correction_idx_s2,_, _ =compute_batch_loss(torch.squeeze(mlabel_cfeats_s2), torch.squeeze(mlabels_u_w_cm2),cfg, all_class_delta_rel, torch.squeeze( mlabels_u_s2))
                    _, mlabel_correction_idx_u_w,_, _ =compute_batch_loss(torch.squeeze(c1s_u_w), torch.squeeze(mlabels_u_w_fp),cfg,all_class_delta_rel,torch.squeeze(torch.zeros(mlabels_u_w_fp.shape)))
                #multiloss_s1, correction_idx_s1,TP_ratio_s1, FN_ratio_s1 = compute_batch_loss(torch.squeeze(classifier_feats_s1), torch.squeeze(mlabels_u_w_cm1),cfg, all_class_delta_rel, torch.squeeze( mlabels_u_s1))

                multiloss_s1=F.binary_cross_entropy_with_logits(torch.squeeze(classifier_feats_s1), torch.squeeze(mlabels_u_w_cm1), reduction='none')#[40401, 19]
                multiloss_s1[mlabel_correction_idx_s1[0], mlabel_correction_idx_s1[1]]=0
                multiloss_s1 = multiloss_s1.mean()
             
                multiloss_s2=F.binary_cross_entropy_with_logits(torch.squeeze(classifier_feats_s2), torch.squeeze(mlabels_u_w_cm2), reduction='none')#[40401, 19]
                multiloss_s2[mlabel_correction_idx_s2[0], mlabel_correction_idx_s2[1]]=0
                multiloss_s2 = multiloss_s2.mean()

                multiloss_u_w_fp=F.binary_cross_entropy_with_logits(torch.squeeze(c1s_u_w_fp), torch.squeeze(mlabels_u_w_fp), reduction='none')#[40401, 19]
                multiloss_u_w_fp[mlabel_correction_idx_u_w[0], mlabel_correction_idx_u_w[1]]=0
                multiloss_u_w_fp = multiloss_u_w_fp.mean()

                for _i_ in range(mlabel_correction_idx_s1[0].cpu().detach().shape[0]):
                    box = mlabel_correction_idx_s1[0].cpu().detach()[_i_].item()
                    cons_class = mlabel_correction_idx_s1[1].cpu().detach()[_i_].item()
                    u_patch_x=int(temp_map[box,0].item())
                    v_patch_x=int(temp_map[box,1].item())
                    u_patch_y=int(temp_map[box,2].item())
                    v_patch_y=int(temp_map[box,3].item())  
                    class_u_w_cutmixed1[:,u_patch_x:v_patch_x, u_patch_y:v_patch_y][mask_u_w_cutmixed1[:,u_patch_x:v_patch_x, u_patch_y:v_patch_y]==cons_class]=200
                for _i_ in range(mlabel_correction_idx_s2[0].cpu().detach().shape[0]):
                    box = mlabel_correction_idx_s2[0].cpu().detach()[_i_].item()
                    cons_class = mlabel_correction_idx_s2[1].cpu().detach()[_i_].item()
                    u_patch_x=int(temp_map[box,0].item())
                    v_patch_x=int(temp_map[box,1].item())
                    u_patch_y=int(temp_map[box,2].item())
                    v_patch_y=int(temp_map[box,3].item())  
                    class_u_w_cutmixed2[:,u_patch_x:v_patch_x, u_patch_y:v_patch_y][mask_u_w_cutmixed2[:,u_patch_x:v_patch_x, u_patch_y:v_patch_y]==cons_class]=200                    
                for _i_ in range(mlabel_correction_idx_u_w[0].cpu().detach().shape[0]):
                    box = mlabel_correction_idx_u_w[0].cpu().detach()[_i_].item()
                    cons_class = mlabel_correction_idx_u_w[1].cpu().detach()[_i_].item()
                    u_patch_x=int(temp_map[box,0].item())
                    v_patch_x=int(temp_map[box,1].item())
                    u_patch_y=int(temp_map[box,2].item())
                    v_patch_y=int(temp_map[box,3].item())  
                    class_u_w_fp[:,u_patch_x:v_patch_x, u_patch_y:v_patch_y][mask_u_w[:,u_patch_x:v_patch_x, u_patch_y:v_patch_y]==cons_class]=200
                #print("--- %s seconds ---" % (time.time() - start_time))
                ignore_mask[class_u_w_fp==200]=255
                loss_u_w_fp = criterion_u(pred_u_w_fp, mask_u_w)
                loss_u_w_fp = loss_u_w_fp * (ignore_mask != 255)
                loss_u_w_fp = loss_u_w_fp.sum() / (ignore_mask != 255).sum().item()
                ignore_mask_cutmixed1[class_u_w_cutmixed1==200]=255
                ignore_mask_cutmixed2[class_u_w_cutmixed2==200]=255
                loss_u_s1 = criterion_u(pred_u_s1, mask_u_w_cutmixed1)
                loss_u_s1 = loss_u_s1 * (ignore_mask_cutmixed1 != 255)
                loss_u_s1 = loss_u_s1.sum() / (ignore_mask_cutmixed1 != 255).sum().item()
                
                loss_u_s2 = criterion_u(pred_u_s2, mask_u_w_cutmixed2)
                loss_u_s2 = loss_u_s2 * (ignore_mask_cutmixed2 != 255)
                loss_u_s2 = loss_u_s2.sum() / (ignore_mask_cutmixed2 != 255).sum().item()
                loss = (loss_x+loss_multi_x + loss_u_s1 * 0.25 + loss_u_s2 * 0.25 + loss_u_w_fp * 0.5 + multiloss_s1 * 0.1 + multiloss_s2 * 0.1 + multiloss_u_w_fp * 0.1 ) / 3.3

                torch.distributed.barrier()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss.update(loss.item())
                total_loss_x.update(loss_x.item())
                total_loss_multi.update(loss_multi_x.item())
                total_loss_s.update((loss_u_s1.item() + loss_u_s2.item()) / 2.0)
                total_loss_w_fp.update(loss_u_w_fp.item())
                total_loss_multi_unlab.update((multiloss_s1.item() + multiloss_s2.item() + multiloss_u_w_fp.item())/3.0)

                mask_ratio = ((conf_u_w >= cfg['conf_thresh']) & (ignore_mask != 255)).sum().item() / \
                    (ignore_mask != 255).sum()
                total_mask_ratio.update(mask_ratio.item())
            
            iters = epoch * len(trainloader_u) + i
            lr = cfg['lr'] * (1 - iters / total_iters) ** 0.9
#            epoch_based_class_unconf={}
#             for _class_ in range(cfg['nclass']):
#                  epoch_based_class_unconf[_class_]=total_all_class_labeled_unconf[_class_].avg
#             if(epoch_change==1):
#                 epoch_change=0
# #                 if epoch > 0:
#                 for _class_ in range(cfg['nclass']):
#                     epoch_class_unconf = epoch_based_class_unconf[_class_]
#                     if(epoch_class_unconf > all_class_max_labeled_unconf[_class_]):
#                         all_class_max_labeled_unconf[_class_]=epoch_class_unconf
#                     all_class_labeled_unconf[_class_].update(epoch_class_unconf)
#                     if(epoch > 0):
#                         max_val = all_class_max_labeled_unconf[_class_]                        
#                         avg_val = all_class_labeled_unconf[_class_].avg
#                         last_val = all_class_labeled_unconf[_class_].val
#                         if(max_val>avg_val):
#                             if(max_val>last_val):
#                                 all_class_delta_rel[_class_]+=cfg['delta_rel']*((max_val-last_val)/max_val)**1.5
            optimizer.param_groups[0]["lr"] = lr
            optimizer.param_groups[1]["lr"] = lr * cfg['lr_multi']            
            if rank == 0:
                writer.add_scalar('train/loss_all', loss.item(), iters)
                writer.add_scalar('train/loss_x', loss_x.item(), iters)
                writer.add_scalar('train/loss_multi_x', loss_multi_x.item(),iters)
                if(epoch>0):
                    writer.add_scalar('train/loss_multi_unlab', (multiloss_s1.item() + multiloss_s2.item() + multiloss_u_w_fp.item())/3.0,iters)
                else:
                    writer.add_scalar('train/loss_multi_unlab', 0,iters)
                if(epoch>0):
                    writer.add_scalar('train/loss_s', (loss_u_s1.item() + loss_u_s2.item()) / 2.0, iters)
                else:
                    writer.add_scalar('train/loss_s', 0, iters)
                if(epoch>0):
                    writer.add_scalar('train/loss_w_fp', loss_u_w_fp.item(), iters)
                else:
                    writer.add_scalar('train/loss_w_fp',0, iters)
                if(epoch>0):
                    writer.add_scalar('train/mask_ratio', mask_ratio, iters) 
                else:
                    writer.add_scalar('train/mask_ratio', 0, iters)
                if(epoch>0):
                    for _class_ in range(cfg['nclass']):
                        writer.add_scalar('train/delta_'+str(_class_), all_class_delta_rel[_class_], iters)
                else:
                    for _class_ in range(cfg['nclass']):
                        writer.add_scalar('train/delta_'+str(_class_), 0, iters)                    
                ##################################################################################################prantik
#                 writer.add_scalar('train/loss_multi_s1', loss_multi_s1.item(),iters)
#                 writer.add_scalar('train/loss_multi_s2', loss_multi_s2.item(),iters)
            
            if (i % (len(trainloader_u) // 8) == 0) and (rank == 0):
                if(epoch==0):
                    logger.info('Iters: {:}, Total loss: {:.3f}, Loss x: {:.3f},loss multi: {:.3f}'.format(i, total_loss.avg, total_loss_x.avg,total_loss_multi.avg))
                else:
                    logger.info('Iters: {:}, Total loss: {:.3f}, Loss x: {:.3f},loss multi: {:.3f}, Loss s: {:.3f}, Loss w_fp: {:.3f}, Mask ratio: {:.3f},  loss multi_unlab: {:.3f}'.format(i, total_loss.avg, total_loss_x.avg,total_loss_multi.avg,total_loss_s.avg,total_loss_w_fp.avg, total_mask_ratio.avg,total_loss_multi_unlab.avg))
        for _class_ in range(cfg['nclass']):
            epoch_class_unconf = total_all_class_labeled_unconf[_class_].avg
            if(epoch_class_unconf > all_class_max_labeled_unconf[_class_]):
                all_class_max_labeled_unconf[_class_]=epoch_class_unconf
                all_class_max_labeled_unconf_epoch[_class_]=epoch
            all_class_labeled_unconf[_class_].update(epoch_class_unconf)
            if(epoch > 0):
                max_val = all_class_max_labeled_unconf[_class_]                        
                avg_val = all_class_labeled_unconf[_class_].max
                last_val = all_class_labeled_unconf[_class_].val
                if(max_val>avg_val):
                    if(max_val>last_val):
                        #print(all_class_delta_rel[_class_])
                        all_class_delta_rel[_class_]+=(((max_val-last_val)/max_val)**1.5)*(cfg['delta_rel'])**(((epoch-all_class_max_labeled_unconf_epoch[_class_])/cfg['epochs'])**1)
                        #all_class_delta_rel[_class_]+=(((max_val-last_val)/max_val)**1.5)*(cfg['delta_rel'])
                        #print(all_class_delta_rel[_class_])
                        #print(aaa)

            #print(epoch)
            #print(total_TP_ratio_s1)
        #print(all_class_delta_rel)
        #print(aaa)

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
            torch.save(checkpoint, os.path.join(args.save_path, 'latest'+'_'+str(epoch)+'.pth'))
            #torch.save(checkpoint, os.path.join(args.save_path, 'latest.pth'))
            if is_best:
                torch.save(checkpoint, os.path.join(args.save_path, 'best.pth'))


if __name__ == '__main__':
    main()
