import torch
from torch import nn
import torch.nn.functional as F

       
class Poolinglabel(torch.nn.Module):
    def __init__(self, all_class=19):
        super(Poolinglabel, self).__init__()
        self.batchnorm_conv1 = nn.Sequential(
                nn.MaxPool2d( kernel_size=3, stride=2, padding=1),
                nn.MaxPool2d( kernel_size=3, stride=1, padding=1),
                nn.MaxPool2d( kernel_size=3, stride=1, padding=1),
                )
        self.batchnorm_maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.batchnorm_layer_1 = nn.Sequential(
                nn.MaxPool2d( kernel_size=3, stride=1, padding=1),
                nn.MaxPool2d( kernel_size=3, stride=1, padding=1),
                nn.MaxPool2d( kernel_size=3, stride=1, padding=1),
                )
        #self.rec_maxpool = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)        
        self.all_class=all_class
        
    def forward(self, x):
        class_list = list(range(self.all_class)) 
        mask_patches = torch.stack([x==i for i in class_list], dim=1)
        mask_patches = mask_patches.float()
        bn_conv1 = self.batchnorm_conv1(mask_patches)#torch.Size([1, 19, 401, 401])
        bn_conv1 = self.batchnorm_maxpool(bn_conv1)#([1, 19, 201, 201])
        bn_layer1 = self.batchnorm_layer_1(bn_conv1) 
        #bn_layer1 = self.rec_maxpool(bn_layer1) 
        bn_layer1 = bn_layer1.type(torch.cuda.HalfTensor)
        return bn_layer1

def calculate_receptive_field(u,v):
    #remeber to put the params below in reverse
#     k = [17,3,3,3,3,3,3,3]
#     s = [1,1,1,1,2,1,1,2]
#     p =[8,1,1,1,1,1,1,1]
    k = [3,3,3,3,3,3,3]
    s = [1,1,1,2,1,1,2]
    p =[1,1,1,1,1,1,1]
    for i in range(len(k)):
        u = -p[i]+(u*s[i])
        v = -p[i]+(v*s[i])+k[i]-1
    return u,v
# def calculate_receptive_field(u,v):
#     #remeber to put the params below in reverse
#     k = [3,3,3,3,3,3,3]
#     s = [1,1,1,2,1,1,2]
#     p =[1,1,1,1,1,1,1]
#     for i in range(len(k)):
#         u = -p[i]+(u*s[i])
#         v = -p[i]+(v*s[i])+k[i]-1
#     return u,v
