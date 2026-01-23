from typing_extensions import final
import torch
from torch._C import ThroughputBenchmark
import torch.nn.functional as F
import math 

   

'''
loss functions
'''

def loss_an(logits, observed_labels):

    assert torch.min(observed_labels) >= 0
    # compute loss:
    loss_matrix = F.binary_cross_entropy_with_logits(logits, observed_labels, reduction='none')#[40401, 19]
    #corrected_loss_matrix = F.binary_cross_entropy_with_logits(logits, torch.logical_not(observed_labels).float(), reduction='none')#[40401, 19]
    #return loss_matrix, corrected_loss_matrix
    return loss_matrix


'''
top-level wrapper
'''

def compute_batch_loss(preds, label_vec,config,all_class_delta_rel, gt_mask): # "preds" are actually logits (not sigmoid activated !)#[40401, 19]) ([40401, 19]
     
    assert preds.dim() == 2
    #print(label_vec.shape, gt_mask.shape)

    
    batch_size = int(preds.size(0))
    num_classes = int(preds.size(1))
    
    unobserved_mask = (label_vec == 1)
    unobserved_mask_negative = (label_vec == 0)
    
    # compute loss for each image and class:
    #loss_matrix, corrected_loss_matrix = loss_an(preds, label_vec.clip(0))#[40401, 19]) ([40401, 19]
    loss_matrix = loss_an(preds, label_vec.clip(0))#[40401, 19]) ([40401, 19]

    correction_idx = None

    if (config['mod_scheme'] == 'LL-Cp'):
        k = math.ceil(batch_size * num_classes * config['delta_rel'])        
    else:
        k = math.ceil(batch_size * num_classes * (1-config['clean_rate']))#check
        
    topk_whole = torch.topk(loss_matrix.flatten(), k)
    unobserved_loss = unobserved_mask.bool() * loss_matrix
    unobserved_loss_negative = unobserved_mask_negative.bool() * loss_matrix
    zero_loss_matrix = torch.zeros_like(loss_matrix)
    all_correction_idx_0=[]
    all_correction_idx_1=[]
    ################################################################
    positive_GT = torch.zeros(gt_mask.shape).cuda()
    positive_GT[gt_mask==1] =1
    negative_GT = torch.zeros(gt_mask.shape).cuda()
    negative_GT[gt_mask==0] =1            
    positive_Pred = torch.zeros(label_vec.shape).cuda()
    positive_Pred[label_vec==1] =1
    negative_Pred = torch.zeros(label_vec.shape).cuda()
    negative_Pred[label_vec==0]=1
    #print(positive_Pred.shape, positive_GT.shape)
    FN = positive_Pred*negative_GT#[40401, 19]
    #epoch_FN +=torch.squeeze(FN).detach().cpu()
    TP = positive_Pred * positive_GT#[40401, 19]
    #epoch_TP +=torch.squeeze(TP).detach().cpu()
    TN = negative_Pred * negative_GT#[40401, 19]
    #epoch_TN +=torch.squeeze(TN).detach().cpu()
    FP = negative_Pred *positive_GT#[40401, 19]
    ########################################################################
    for _class_ in range(config['nclass']):
            considered_class = torch.zeros(loss_matrix.shape).cuda()
            considered_class[:,_class_]=1
            if (config['mod_scheme'] == 'LL-Cp'): 
                k_new = math.ceil(torch.sum(considered_class* unobserved_mask).item() *  all_class_delta_rel[_class_])
            else:
                k_new = math.ceil(torch.sum(considered_class*unobserved_mask).item() *  (1-config['clean_rate']))#check

            #print(_class_,torch.sum(considered_class).item(), k_new)
            if(k_new !=0):
                topk = torch.topk((considered_class.bool()*unobserved_loss).flatten(), k_new)
                topk_lossvalue = topk.values[-1]
                loss_matrix = torch.where((considered_class.bool()*unobserved_loss) < topk_lossvalue, loss_matrix, zero_loss_matrix)
                correction_idx = torch.where((considered_class.bool()*unobserved_loss) > topk_lossvalue)
                all_correction_idx_0.append(correction_idx[0])
                all_correction_idx_1.append(correction_idx[1])
    #print(sasasas)        
    all_correction_idx_0 = torch.cat(all_correction_idx_0)
    all_correction_idx_1 = torch.cat(all_correction_idx_1)

    topk_lossvalue = topk_whole.values[-1]
    loss_matrix = torch.where(unobserved_loss_negative < topk_lossvalue, loss_matrix, zero_loss_matrix)
#     try:
#         topk = torch.topk(unobserved_loss.flatten(), k)
#         topk_negative = torch.topk(unobserved_loss_negative.flatten(), k)
#     except:
#         print("check here", k,unobserved_loss.flatten().shape, batch_size,num_classes,config['clean_rate'])
#         print(sasassas)
#     topk_lossvalue = topk.values[-1]
#     topk_negative_lossvalue = topk_negative.values[-1]
#     correction_idx = torch.where(unobserved_loss > topk_lossvalue)
#     if config['mod_scheme'] in ['LL-Ct', 'LL-Cp']:
#         final_loss_matrix = torch.where(unobserved_loss < topk_lossvalue, loss_matrix, corrected_loss_matrix)
#     else:
        
#         final_loss_matrix = torch.where(unobserved_loss < topk_lossvalue, loss_matrix, zero_loss_matrix)
#         final_loss_matrix = torch.where(unobserved_loss_negative < topk_negative_lossvalue, final_loss_matrix, zero_loss_matrix)
    with torch.no_grad():
        TP_ORG_SUM = torch.sum(TP, dim=0)#19
        TP[loss_matrix==0]=0
        TP_ratio = torch.div((TP_ORG_SUM-torch.sum(TP,dim=0)),TP_ORG_SUM)#ratio of TP removed by threshold
        TP_ratio = torch.nan_to_num(TP_ratio)
        TP_ratio = TP_ratio.detach()
        

        FN_ORG_SUM = torch.sum(FN, dim=0)
        FN[loss_matrix==0]=0
        FN_ratio = torch.div((FN_ORG_SUM-torch.sum(FN,dim=0)),FN_ORG_SUM)#ratio of FN removed by threshold
        FN_ratio = torch.nan_to_num(FN_ratio)
        FN_ratio = FN_ratio.detach()
    main_loss = loss_matrix.mean()
    return main_loss, (all_correction_idx_0,all_correction_idx_1), TP_ratio, FN_ratio