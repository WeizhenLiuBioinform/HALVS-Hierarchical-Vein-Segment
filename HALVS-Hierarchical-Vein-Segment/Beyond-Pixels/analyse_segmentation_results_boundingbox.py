
###################analysing the segmentation_results as numpy
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
from supervised import get_segmentation_bbox
from util.classes import CLASSES
from util.ohem import ProbOhemCrossEntropy2d
from util.utils import count_params, init_log, AverageMeter
from util.dist_helper import setup_distributed
import pickle
import numpy as np
from model.semseg.maxpool_label import Poolinglabel
from model.semseg.maxpool_label import calculate_receptive_field

#############################save best epoch
parser = argparse.ArgumentParser(description='Revisiting Weak-to-Strong Consistency in Semi-Supervised Semantic Segmentation')
parser.add_argument('--config', type=str, required=True)
parser.add_argument('--labeled-id-path', type=str, required=True)
parser.add_argument('--unlabeled-id-path', type=str, required=True)
parser.add_argument('--save-path', type=str, required=True)
parser.add_argument('--local-rank', default=0, type=int)
parser.add_argument('--port', default=None, type=int)



def main():
    args = parser.parse_args()

    cfg = yaml.load(open(args.config, "r"), Loader=yaml.Loader)


    rank, world_size = setup_distributed(port=args.port)

    if rank == 0:        
        os.makedirs(args.save_path, exist_ok=True)
    
    cudnn.enabled = True
    cudnn.benchmark = True

    model = DeepLabV3Plus(cfg)
    optimizer = SGD([{'params': model.backbone.parameters(), 'lr': cfg['lr']},
                    {'params': [param for name, param in model.named_parameters() if 'backbone' not in name],
                    'lr': cfg['lr'] * cfg['lr_multi']}], lr=cfg['lr'], momentum=0.9, weight_decay=1e-4)
    


    local_rank = int(os.environ["LOCAL_RANK"])
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model.cuda()

    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], broadcast_buffers=False,
                                                      output_device=local_rank, find_unused_parameters=False)
    valset = SemiDataset(cfg['dataset'], cfg['data_root'], 'val')
    trainset_u = SemiDataset(cfg['dataset'], cfg['data_root'], 'train_u',cfg['crop_size'], args.unlabeled_id_path, unlab_val= 'train_u_val')
    trainsampler_u = torch.utils.data.distributed.DistributedSampler(trainset_u)
    trainloader_u = DataLoader(trainset_u, batch_size=1,
                               pin_memory=True, num_workers=1, drop_last=True, sampler=trainsampler_u)

#     trainset_l = SemiDataset(cfg['dataset'], cfg['data_root'], 'train_l',
#                              cfg['crop_size'], args.unlabeled_id_path)


#     #valset = SemiDataset(cfg['dataset'], cfg['data_root'], 'val')

#     trainsampler_l = torch.utils.data.distributed.DistributedSampler(trainset_l)
#     trainloader_l = DataLoader(trainset_l, batch_size=1,
#                                pin_memory=True, num_workers=1, drop_last=True, sampler=trainsampler_l)
    valsampler = torch.utils.data.distributed.DistributedSampler(valset)
    valloader = DataLoader(valset, batch_size=1, pin_memory=True, num_workers=1,
                           drop_last=False, sampler=valsampler)


    #criterion_l = torch.nn.CrossEntropyLoss(reduction='none',ignore_index=255).cuda(local_rank)
    criterion_l=ProbOhemCrossEntropy2d(reduction='none',**cfg['criterion']['kwargs']).cuda(local_rank)
    #MODEL_NAME = 'best.pth'
    #PATH_SAVE='./output_pics_'+MODEL_NAME.split('.')[0]
    #calcualate the size of the original image
    t_img, t_mask= trainset_u[0]#[3, 321, 321],[321, 321]
    org_u_left=0
    org_v_right= t_mask.shape[1]-1
    pred_u_left =0
    pred_v_right=80
    list_receptive_field=[]
    for st in range(pred_v_right+1):
        list_receptive_field.append([calculate_receptive_field(st,st)])
    receptive_pad_left = org_u_left-list_receptive_field[0][0][0]
    receptive_pad_right = list_receptive_field[-1][0][1]-org_v_right
    receptive_kernel_size = list_receptive_field[0][0][1]-list_receptive_field[0][0][0]
    receptive_stride = list_receptive_field[1][0][0]-list_receptive_field[0][0][0]
    temp_map = torch.zeros(4,81,81)
    for _i_ in range(81):
        for _j_ in range(81):
            _u_x,_v_x = calculate_receptive_field(_i_,_i_)
            if(_u_x<0):
                _u_x=0
            if(_u_x>=321):
                _u_x=320
            if(_v_x<0):
                _v_x=0
            if(_v_x>=321):
                _v_x=320            
            temp_map[0,_i_,_j_]=_u_x
            temp_map[1,_i_,_j_]=_v_x
            _u_y,_v_y = calculate_receptive_field(_j_,_j_)
            if(_u_y<0):
                _u_y=0
            if(_u_y>=321):
                _u_y=320
            if(_v_y<0):
                _v_y=0
            if(_v_y>=321):
                _v_y=320            
            temp_map[2,_i_,_j_]=_u_y
            temp_map[3,_i_,_j_]=_v_y

    temp_map = temp_map.view(temp_map.shape[0],-1)
    temp_map = temp_map.permute(1,0)#[40401, 4]

    ALL_FILES=[]
#     for r,d,f in os.walk(args.save_path):
#         for file in f:
#             if('ipynb' not in file):
#                 if('pth' in file):
#                     ALL_FILES.append(file)
    
    ALL_FILES = ['best.pth']###############only if u. want the best model
    for MODEL_NAME in ALL_FILES:
        if not os.path.exists('./all_output_bboxes'):
                os.makedirs('./all_output_bboxes')
        PATH_SAVE='./all_output_bboxes/output_pics_'+MODEL_NAME.split('.')[0]        
        if not os.path.exists(PATH_SAVE):
                os.makedirs(PATH_SAVE)
        checkpoint = torch.load(os.path.join(args.save_path, MODEL_NAME))
        model.load_state_dict(checkpoint['model'])
        model.cuda()
        model.eval()
        print("MODEL_NAME: ",os.path.join(args.save_path, MODEL_NAME))
        eval_mode = 'sliding_window' if cfg['dataset'] == 'cityscapes' else 'original'
        get_segmentation_bbox(model, trainloader_u, eval_mode, cfg, PATH_SAVE, temp_map)



if __name__ == '__main__':
    main()

