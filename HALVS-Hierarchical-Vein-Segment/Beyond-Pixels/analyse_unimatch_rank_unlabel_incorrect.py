
###################analysing the rank of the predictions fo the labeled images
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
from util.dist_helper import setup_distributed
import pickle
import numpy as np
#torch.set_printoptions(threshold=10000)
#PATH_SAVE = '/nfs/bigcortex.cs.stonybrook.edu/add_disk0/ironman/Unimatch_ranking'
#PATH_SAVE = '/nfs/bigcone.cs.stonybrook.edu/add_disk0/ironman/Unimatch_ranking'
#PATH_SAVE = '/nfs/bigrod.cs.stonybrook.edu/add_disk0/ironman/Unimatch_ranking'
PATH_SAVE='./unlabeled_images_incorrect_ranking'
parser = argparse.ArgumentParser(description='Revisiting Weak-to-Strong Consistency in Semi-Supervised Semantic Segmentation')
parser.add_argument('--config', type=str, required=True)
parser.add_argument('--labeled-id-path', type=str, required=True)
parser.add_argument('--unlabeled-id-path', type=str, required=True)
parser.add_argument('--save-path', type=str, required=True)
parser.add_argument('--local_rank', default=0, type=int)
parser.add_argument('--port', default=None, type=int)
parser.add_argument('--startepoch', default=0, type=int)
parser.add_argument('--endepoch', default=0, type=int)

def return_stats(class_top_k_indices):
    #print(class_top_k_indices.shape)##(36947, 2)
    temp_dict={}
    for _i_ in range(class_top_k_indices.shape[0]):
        temp = class_top_k_indices[_i_,:]#(2,)
#         if (considered_class == temp[0]):
        temp = tuple(temp)
        if(temp_dict.get(temp) is None):
            temp_dict[temp]=1
        temp_dict[temp]+=1
    return temp_dict


def main():
    ########################################################################change top K
    K_CONSIDER=[2,3,4,5,6,7,8,9,10,11,12]
    CONFIDENCE = [[0.95,1],[0.90,0.95],[0.85,0.90],[0.8,0.85],[0.75,0.8],[0.7,0.75],[0.65,0.7],[0.6,0.65],[0.55,0.6],[0.5,0.55]]
    NOT_CONSIDER=100#makes the noncosidered index and corresponding values
    CLASS_0=50
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

    #only for the labeled images
    trainset_l = SemiDataset(cfg['dataset'], cfg['data_root'], 'train_l',
                             cfg['crop_size'], args.unlabeled_id_path)


    #valset = SemiDataset(cfg['dataset'], cfg['data_root'], 'val')

    trainsampler_l = torch.utils.data.distributed.DistributedSampler(trainset_l)
    trainloader_l = DataLoader(trainset_l, batch_size=1,
                               pin_memory=True, num_workers=1, drop_last=True, sampler=trainsampler_l)#10490


    print(cfg['criterion']['kwargs'])
    #criterion_l = torch.nn.CrossEntropyLoss(reduction='none',ignore_index=255).cuda(local_rank)
    criterion_l=ProbOhemCrossEntropy2d(reduction='none',**cfg['criterion']['kwargs']).cuda(local_rank)
    if not os.path.exists(PATH_SAVE):
        os.makedirs(PATH_SAVE)
    for epoch in range(args.startepoch,args.endepoch):
        print("epoch ", str(epoch))
        PATH_SAVE_FOLDER = PATH_SAVE+'/'+str(epoch)
        if not os.path.exists(PATH_SAVE_FOLDER):
            os.makedirs(PATH_SAVE_FOLDER)
        print(os.path.join(args.save_path, str(epoch)+'_latest.pth'))
        checkpoint = torch.load(os.path.join(args.save_path, str(epoch)+'_latest.pth'))
        model.load_state_dict(checkpoint['model'])
        model.cuda()
        model.eval()
        
        for _k_ in K_CONSIDER:
            final_dict={}
            final_dict[_k_]={}
            for b_conf, e_conf in CONFIDENCE:
                final_dict[_k_][tuple([b_conf, e_conf])]={}
                for i, ((img_x, mask_x)) in enumerate(trainloader_l):
                    img_x, mask_x = img_x.cuda(), mask_x.cuda()#[1, 3, 321, 321], [1, 321, 321]
                    pred_x = model(img_x).detach()#[1, 21, 321, 321]
                    softmax_pred_x = pred_x.softmax(dim=1)#[1, 21, 321, 321]
                    softmax_pred_x = torch.squeeze(softmax_pred_x)#[21, 321, 321]
                    softmax_pred_x_top_k_values,softmax_pred_x_top_k_indices  =torch.topk(softmax_pred_x,softmax_pred_x.shape[0],dim=0)#[21, 321, 321]) ([21, 321, 321]                    
                    pred_conf_x = pred_x.softmax(dim=1).max(dim=1)[0]#[1, 321, 321]
                    pred_mask_x = pred_x.argmax(dim=1)#[1, 321, 321]
                    loss_x = criterion_l(pred_x, mask_x)#torch.Size([1, 801, 801])
                    pred_mask_x[pred_mask_x==0]=CLASS_0
                    mask_x[mask_x==0]= CLASS_0
        ####################################################################################################################
                    #####considered pseydo labels based on confidence
                    considered_psseudo_labels = torch.zeros(pred_conf_x.shape).cuda()
                    considered_psseudo_labels[pred_conf_x<e_conf]=1
                    considered_psseudo_labels[pred_conf_x<b_conf]=0
                    ########pseudo labels above threshold
                    pseudo_mask_x = considered_psseudo_labels*pred_mask_x #torch.Size([1, 801, 801])
                    #pseudo_mask_x = pred_mask_x.clone()#[1, 321, 321]
                    #pseudo_mask_x[pseudo_mask_x==0]=NOT_CONSIDER
        ######################################################################################################################
                    incorrect_pseudo_labels = torch.zeros(pseudo_mask_x.shape).cuda()
                    incorrect_pseudo_labels[pseudo_mask_x!=mask_x]=1 
                    incorrect_pseudo_labels[pseudo_mask_x==0]=0
#                     test_1 = incorrect_pseudo_labels*pred_conf_x
#                     print(torch.unique(test_1))
#                     print(torch.unique(incorrect_pseudo_labels.squeeze()*softmax_pred_x_top_k_values[0,:,:]))
#                     print(torch.min(torch.unique(test_1)))
#                     print(torch.min(torch.unique(incorrect_pseudo_labels.squeeze()*softmax_pred_x_top_k_values[0,:,:])))
#                     print(torch.max(torch.unique(incorrect_pseudo_labels.squeeze()*softmax_pred_x_top_k_values[0,:,:])))
#                     print(aaa)
                    PL_CLASSES= list(torch.unique(pseudo_mask_x))
                    if (0 in PL_CLASSES):
                        PL_CLASSES.remove(0)
#                     print("check PL class", PL_CLASSES)
                    for considered_class in PL_CLASSES:
                        if(considered_class < 100):
                            class_incorrect_pseudo_labels = incorrect_pseudo_labels.clone()#[1, 321, 321]
                            class_incorrect_pseudo_labels[pseudo_mask_x!=considered_class]=0
                            if(torch.sum(class_incorrect_pseudo_labels).item()>0):
                                if considered_class.item() not in final_dict[_k_][tuple([b_conf, e_conf])].keys():
                                    final_dict[_k_][tuple([b_conf, e_conf])][considered_class.item()]={}
                                test_3 = class_incorrect_pseudo_labels.clone()#[1, 321, 321]#only consider the pixels with the considered class in the pseudo labels
#                                 print("PL considered class",considered_class, torch.unique(test_3*pseudo_mask_x))
#                                 print(torch.unique(test_3.squeeze()*softmax_pred_x_top_k_indices[0,:,:]))
#                                 print(torch.unique(test_3.squeeze()*softmax_pred_x_top_k_values[0,:,:]))
                                #print(xxx)
                                GT_incorrect_pseudo_labels = mask_x.clone()#[1, 321, 321]
                                GT_incorrect_pseudo_labels[test_3==0]=0#only consider the GT pixels for the considered class in the pseudo labels
                                GT_CLASSES = list(torch.unique(GT_incorrect_pseudo_labels))
                                if(0 in GT_CLASSES):
                                    GT_CLASSES.remove(0)
#                                 print("GT CLASSES", GT_CLASSES)
                                for considered_GT_class in GT_CLASSES:
                                    if considered_GT_class.item() not in final_dict[_k_][tuple([b_conf, e_conf])][considered_class.item()].keys():
                                        final_dict[_k_][tuple([b_conf, e_conf])][considered_class.item()][considered_GT_class.item()]={}                                
#                                     print(considered_class, considered_GT_class)
#                                     print(torch.unique(test_3), torch.unique()
#                                     print(torch.sum(test_3))
#                                     print(torch.unique(test_3.squeeze()*softmax_pred_x_top_k_indices[0,:,:]))
#                                     aa = torch.zeros(softmax_pred_x_top_k_indices[0,:,:].shape)
#                                     aa[ (test_3.squeeze()*softmax_pred_x_top_k_indices[0,:,:])==considered_class]=1
#                                     print(torch.sum(aa))
#                                     print(aaa)
                                                                                                                  
                                    class_GT = torch.zeros(GT_incorrect_pseudo_labels.shape).cuda()#[1, 321, 321]
                                    class_GT[GT_incorrect_pseudo_labels==considered_GT_class]=1#only consider the pixels that have the corresponding consideredGTclass and consideredclass 
#                                     print("considered GT class ", considered_GT_class)
#                                     print(torch.unique(class_GT*mask_x))
#                                     print(torch.unique(class_GT* pred_mask_x))
#                                     print(torch.unique(class_GT * softmax_pred_x_top_k_indices[0,:,:]))
#                                     print(torch.unique(class_GT * softmax_pred_x_top_k_values[0,:,:]))
#                                     print(aaa)
                                    class_top_k_indices= softmax_pred_x_top_k_indices.clone()
                                    class_top_k_values = softmax_pred_x_top_k_values.clone()
                                    class_top_k_indices=class_top_k_indices[0:_k_,:,:]#[2, 321, 321]
                                    class_top_k_values = class_top_k_values[0:_k_,:,:]
                                    class_GT = class_GT.repeat(_k_,1,1)#[2, 321, 321]
                                    class_top_k_indices[class_GT==0]= NOT_CONSIDER
                                    class_top_k_values[class_GT==0]= NOT_CONSIDER
                                    class_top_k_indices = class_top_k_indices.view(class_top_k_indices.shape[0],-1)#[2, 103041]
                                    class_top_k_indices = class_top_k_indices.permute(1,0)
                                    class_top_k_indices = class_top_k_indices.detach().cpu().numpy()
                                    class_top_k_values = class_top_k_values.view(class_top_k_values.shape[0],-1)
                                    class_top_k_values = class_top_k_values.permute(1,0)
                                    class_top_k_values = class_top_k_values.detach().cpu().numpy()                                
                                    rank_firstindex = class_top_k_indices[:,0]
                                    locs = np.squeeze(np.argwhere(rank_firstindex==NOT_CONSIDER), axis=1)
                                    class_top_k_indices = np.delete(class_top_k_indices, locs.tolist(), 0) 
                                    class_top_k_indices = np.int8(class_top_k_indices)
                                    class_top_k_values = np.delete(class_top_k_values, locs.tolist(), 0)
                                    local_stats = return_stats(class_top_k_indices)
                                    for _key_ in local_stats.keys():
                                        if(final_dict[_k_][tuple([b_conf, e_conf])][considered_class.item()][considered_GT_class.item()].get(_key_) is None):
                                            final_dict[_k_][tuple([b_conf, e_conf])][considered_class.item()][considered_GT_class.item()][_key_]=0
                                        final_dict[_k_][tuple([b_conf, e_conf])][considered_class.item()][considered_GT_class.item()][_key_]+=local_stats[_key_]
            with open(PATH_SAVE_FOLDER+'/final_dict_'+str(_k_)+'.txt', 'w') as f:
                for k_key in final_dict.keys():
                    f.write('#####CONSIDER K'+' : '+str(k_key)+'\n')
                    for s_prob, e_prob in final_dict[k_key].keys():
                        f.write('!!!!!!CONSIDER PROB'+' : '+str(s_prob)+' - '+str(e_prob)+'\n')
                        for _class_ in final_dict[k_key][(s_prob,e_prob)].keys():
                            f.write('-----CONSIDER PRED CLASS'+' : '+str(_class_)+'\n')
                            for _gtclass_ in final_dict[k_key][(s_prob,e_prob)][_class_].keys():
                                f.write('-----GT CLASS'+' : '+str(_gtclass_)+'\n')
                                x = sorted(final_dict[k_key][(s_prob,e_prob)][_class_][_gtclass_].items(), key=lambda x:x[1], reverse=True)
                                for line in x:
                                    f.write(str(line[0])+' : '+str(line[1])+'\n')





if __name__ == '__main__':
    main()
