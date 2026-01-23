
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
from supervised import get_segmentation
from util.classes import CLASSES
from util.ohem import ProbOhemCrossEntropy2d
from util.utils import count_params, init_log, AverageMeter
from util.dist_helper import setup_distributed
import pickle
import numpy as np

#############################save best epoch
MODEL_NAME = 'best.pth'
PATH_SAVE='./output_numpy_'+MODEL_NAME.split('.')[0]
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

#     trainset_l = SemiDataset(cfg['dataset'], cfg['data_root'], 'train_l',
#                              cfg['crop_size'], args.unlabeled_id_path)


#     #valset = SemiDataset(cfg['dataset'], cfg['data_root'], 'val')

#     trainsampler_l = torch.utils.data.distributed.DistributedSampler(trainset_l)
#     trainloader_l = DataLoader(trainset_l, batch_size=1,
#                                pin_memory=True, num_workers=1, drop_last=True, sampler=trainsampler_l)
    valsampler = torch.utils.data.distributed.DistributedSampler(valset)
    valloader = DataLoader(valset, batch_size=1, pin_memory=True, num_workers=1,
                           drop_last=False, sampler=valsampler)

    print(cfg['criterion']['kwargs'])
    #criterion_l = torch.nn.CrossEntropyLoss(reduction='none',ignore_index=255).cuda(local_rank)
    criterion_l=ProbOhemCrossEntropy2d(reduction='none',**cfg['criterion']['kwargs']).cuda(local_rank)
    if not os.path.exists(PATH_SAVE):
            os.makedirs(PATH_SAVE)
    checkpoint = torch.load(os.path.join(args.save_path, MODEL_NAME))
    model.load_state_dict(checkpoint['model'])
    model.cuda()
    model.eval()
    print("MODEL_NAME: ",os.path.join(args.save_path, MODEL_NAME))
    eval_mode = 'sliding_window' if cfg['dataset'] == 'cityscapes' else 'original'
    print(eval_mode)
    get_segmentation(model, valloader, eval_mode, cfg, PATH_SAVE)



if __name__ == '__main__':
    main()
