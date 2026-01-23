import model.backbone.resnet as resnet
from model.backbone.xception import xception

import torch
from torch import nn
import torch.nn.functional as F


class DeepLabV3Plus(nn.Module):
    def __init__(self, cfg):
        super(DeepLabV3Plus, self).__init__()


        if 'resnet' in cfg['backbone']:
            self.backbone = resnet.__dict__[cfg['backbone']](pretrained=True, 
                                                             replace_stride_with_dilation=cfg['replace_stride_with_dilation'])

        else:
            assert cfg['backbone'] == 'xception'
            self.backbone = xception(pretrained=True)

        low_channels = 256
        high_channels = 2048

        self.head = ASPPModule(high_channels, cfg['dilations'])

        self.reduce = nn.Sequential(nn.Conv2d(low_channels, 48, 1, bias=False),
                                    nn.BatchNorm2d(48),
                                    nn.ReLU(True))

        self.fuse = nn.Sequential(nn.Conv2d(high_channels // 8 + 48, 256, 3, padding=1, bias=False),
                                  nn.BatchNorm2d(256),
                                  nn.ReLU(True),
                                  nn.Conv2d(256, 256, 3, padding=1, bias=False),
                                  nn.BatchNorm2d(256),
                                  nn.ReLU(True))

        self.classifier = nn.Conv2d(256, cfg['nclass'], 1, bias=True)
#         self.convolution_prantik = nn.Conv2d(256,128,  kernel_size=1, stride=1, bias=False)
#         self.classifier_prantik_1 = nn.Linear(128, 64)#128
#         self.classifier_prantik_2 = nn.Linear(64, 19)
#         self.dropout_prantik = nn.Dropout(0.20)
#####################################################################################################
        self.convolution_prantik = nn.Sequential( ##########################################################################68.29
                                                 nn.Conv2d(256,256,  kernel_size=1, stride=1, bias=False),
                                                 nn.BatchNorm2d(256),
                                                 nn.ReLU(True),
                                                 nn.Conv2d(256,256,  kernel_size=3, padding=1,stride=1, bias=False),
                                                 nn.BatchNorm2d(256),
                                                 nn.ReLU(True),
                                                 nn.Conv2d(256,256,  kernel_size=3, padding=1,stride=1, bias=False),
                                                 nn.BatchNorm2d(256),
                                                 nn.ReLU(True),
                                                 nn.Conv2d(256,128,  kernel_size=1, stride=1, bias=False),
                                                 nn.BatchNorm2d(128),
                                                 nn.ReLU(True),
                                                 nn.Conv2d(128,128,  kernel_size=3, padding=1,stride=1, bias=False),
                                                 nn.BatchNorm2d(128),
                                                 nn.ReLU(True),
                                                 nn.Conv2d(128,128,  kernel_size=3, padding=1,stride=1, bias=False),
                                                 nn.BatchNorm2d(128),
                                                 nn.ReLU(True),
                                                 nn.Conv2d(128,64,  kernel_size=1, stride=1, bias=False),
                                                 nn.BatchNorm2d(64),
                                                 nn.ReLU(True),
                                                 nn.Conv2d(64,64,  kernel_size=3, stride=1,padding=1, bias=False),#####this add makes 68.61
                                                 nn.BatchNorm2d(64),
                                                 nn.ReLU(True)
                                                )
#########################################################################################################
#         self.convolution_prantik_1 = nn.Sequential(
#                                                   nn.Conv2d(256,128,  kernel_size=1, stride=1, bias=False),
#                                                   nn.BatchNorm2d(128),
#                                                   nn.ReLU(True),   
#                                                   nn.Conv2d(128,128,  kernel_size=3, padding=1,stride=1, bias=False),
#                                                   nn.BatchNorm2d(128),
#                                                   nn.ReLU(True),
#                                                   nn.Conv2d(128,512,  kernel_size=1, stride=1, bias=False),
#                                                   nn.BatchNorm2d(512),
                                                  
#         )
#         self.convolution_prantik_identity_1 = nn.Sequential(
#                                                   nn.Conv2d(256,512,  kernel_size=1, stride=1, bias=False),
#                                                   nn.BatchNorm2d(512),
#         )
#         self.convolution_prantik_2 = nn.Sequential(
#                                                   nn.Conv2d(512,128,  kernel_size=1, stride=1, bias=False),
#                                                   nn.BatchNorm2d(128),
#                                                   nn.ReLU(True),   
#                                                   nn.Conv2d(128,128,  kernel_size=3, padding=1,stride=1, bias=False),
#                                                   nn.BatchNorm2d(128),
#                                                   nn.ReLU(True),
#                                                   nn.Conv2d(128,512,  kernel_size=1, stride=1, bias=False),
#                                                   nn.BatchNorm2d(512),
                                                  
#         )
#         self.convolution_prantik_3 = nn.Sequential(
#                                                   nn.Conv2d(512,128,  kernel_size=1, stride=1, bias=False),
#                                                   nn.BatchNorm2d(128),
#                                                   nn.ReLU(True),   
#                                                   nn.Conv2d(128,128,  kernel_size=3, padding=1,stride=1, bias=False),
#                                                   nn.BatchNorm2d(128),
#                                                   nn.ReLU(True),
#                                                   nn.Conv2d(128,512,  kernel_size=1, stride=1, bias=False),
#                                                   nn.BatchNorm2d(512),
                                                  
#         )

#         self.convolution_prantik_final = nn.Sequential(
#                                                   nn.Conv2d(512,64,  kernel_size=1, stride=1, bias=False),
#                                                   nn.BatchNorm2d(64),
#                                                   nn.ReLU(True),                                                        
#         )
#########################################################################################################68.61
        self.classifier_prantik_1 = nn.Linear(64, 32)#128
        self.classifier_prantik_2 = nn.Linear(32, 19)
        self.dropout_prantik = nn.Dropout(0.20)
        #self.classifier_prantik_1_bnorm = nn.BatchNorm1d(num_features=32)
########################################################################################################
#         self.convolution_prantik = nn.Sequential( ##########################################################################68.29
#                                                  nn.Conv2d(256,256,  kernel_size=1, stride=1, bias=False),
#                                                  nn.BatchNorm2d(256),
#                                                  nn.ReLU(True),
#                                                  nn.Conv2d(256,256,  kernel_size=3, padding=1,stride=1, bias=False),
#                                                  nn.BatchNorm2d(256),
#                                                  nn.ReLU(True),
#                                                  nn.Conv2d(256,128,  kernel_size=1, stride=1, bias=False),
#                                                  nn.BatchNorm2d(128),
#                                                  nn.ReLU(True),
#                                                  nn.Conv2d(128,128,  kernel_size=3, padding=1,stride=1, bias=False),
#                                                  nn.BatchNorm2d(128),
#                                                  nn.ReLU(True),
#                                                  nn.Conv2d(128,64,  kernel_size=1, stride=1, bias=False),
#                                                  nn.BatchNorm2d(64),
#                                                  nn.ReLU(True),
#                                                  nn.Conv2d(64,64,  kernel_size=3, stride=1,padding=1, bias=False),#####this add makes 68.61
#                                                  nn.BatchNorm2d(64),
#                                                  nn.ReLU(True),
#                                                  nn.Conv2d(64,32,  kernel_size=1, stride=1, bias=False),#######new
#                                                  nn.BatchNorm2d(32),
#                                                  nn.ReLU(True),
#                                                  nn.Conv2d(32,32,  kernel_size=3, stride=1,padding=1, bias=False),
#                                                  nn.BatchNorm2d(32),
#                                                  nn.ReLU(True)
#                                                 )
#         self.classifier_prantik_2 = nn.Linear(32, 19)

    def forward(self, x, need_fp=False, classify=False, nlabel=0, val_feat=False):
        #print("model1", x.shape)#torch.Size([4, 3, 801, 801])
        h, w = x.shape[-2:]

        feats = self.backbone.base_forward(x)
        c1, c4 = feats[0], feats[-1]
        #print("model3", c1.shape, c4.shape)#torch.Size([2, 256, 201, 201]) torch.Size([2, 2048, 51, 51])

        if need_fp:
            outs = self._decode(torch.cat((c1, nn.Dropout2d(0.5)(c1))),
                                torch.cat((c4, nn.Dropout2d(0.5)(c4))))
            #print("model4", outs.shape)#torch.Size([4, 19, 201, 201])
            outs = F.interpolate(outs, size=(h, w), mode="bilinear", align_corners=True)
            #print("model5", outs.shape)#torch.Size([4, 19, 801, 801])
            out, out_fp = outs.chunk(2)
            #print("model6", out.shape, out_fp.shape)#torch.Size([2, 19, 801, 801]) torch.Size([2, 19, 801, 801])
            #print(asasasas)
            if classify:
################################################################68.61
                #c1_lb = c1[:nlabel]#torch.Size([1, 256, 201, 201])
                #print(torch.cat( (c1,nn.Dropout2d(0.5)(c1)) ).shape)#[4, 256, 201, 201]
                #print(asasasas)
                c1_lb = self.convolution_prantik(torch.cat( (c1,nn.Dropout2d(0.5)(c1))))#[4, 64, 201, 201]
                c1_lb = c1_lb.view(c1_lb.shape[0],c1_lb.shape[1],-1)
                c1_lb = c1_lb.permute(0,2,1)#torch.Size([4, 40401, 64])
                classifier_feats = F.relu(self.classifier_prantik_1(c1_lb))
                classifier_feats = self.dropout_prantik(classifier_feats)
                classifier_feats = self.classifier_prantik_2(classifier_feats)#([4, 40401, 19])  
                classifier_feat, classifier_feat_fp = classifier_feats.chunk(2)#[2, 40401, 19])([2, 40401, 19]
#################################################################
#                 c1_lb_1 = self.convolution_prantik_1(c1_lb)
#                 print(c1_lb_1.shape)
#                 print(aaa)
#                 c1_lb_identity = self.convolution_prantik_identity_1(c1_lb)
#                 c1_lb_1 = F.relu(c1_lb_1+c1_lb_identity)
#                 c1_lb_2 =  self.convolution_prantik_2(c1_lb_1)
#                 c1_lb_2 = F.relu(c1_lb_2+c1_lb_1)
#                 c1_lb_3 =  self.convolution_prantik_3(c1_lb_2)
#                 c1_lb_3 = F.relu(c1_lb_3+c1_lb_2)
#                 c1_lb_3 = self.convolution_prantik_final(c1_lb_3)
#                 c1_lb_3 = c1_lb_3.view(c1_lb_3.shape[0],c1_lb_3.shape[1],-1)
#                 c1_lb_3 = c1_lb_3.permute(0,2,1)
#                 classifier_feat = F.relu(self.classifier_prantik_1(c1_lb_3))
#                 classifier_feat = self.dropout_prantik(classifier_feat)
#                 classifier_feat = self.classifier_prantik_2(classifier_feat)#torch.Size([2, 40401, 19])  
#                 c1_lb = c1[:nlabel]#torch.Size([2, 256, 201, 201])
#                 c1_lb = self.convolution_prantik(c1_lb)
#                 c1_lb = c1_lb.view(c1_lb.shape[0],c1_lb.shape[1],-1)
#                 c1_lb = c1_lb.permute(0,2,1)#torch.Size([2, 40401, 256])
# #                 classifier_feat = F.relu(self.classifier_prantik_1(c1_lb))
# #                 classifier_feat = self.dropout_prantik(classifier_feat)
#                 classifier_feat = self.classifier_prantik_2(c1_lb)#torch.Size([2, 40401, 19])
                return classifier_feat, classifier_feat_fp, out, out_fp
                #return c1[:nlabel],classifier_feat,out, out_fp

            return out, out_fp
        if(val_feat):
            feature,out = self._decode(c1, c4, val_feat=True)
            #print(c1.shape,c4.shape, feature.shape, out.shape)#([2, 256, 201, 201]) ([2, 2048, 51, 51])([2, 256, 201, 201])[2, 19, 201, 201]
            out = F.interpolate(out, size=(h, w), mode="bilinear", align_corners=True)#[2, 19, 801, 801]
            feature = F.interpolate(feature, size=(h, w), mode="bilinear", align_corners=True)#[1, 256, 801, 801]
            return feature,out

        out = self._decode(c1, c4)
        out = F.interpolate(out, size=(h, w), mode="bilinear", align_corners=True)
        if classify:
            c1 = self.convolution_prantik(c1)
            c1 = c1.view(c1.shape[0],c1.shape[1],-1)
            c1 = c1.permute(0,2,1)#torch.Size([2, 40401, 256])
            classifier_feats = F.relu(self.classifier_prantik_1(c1))
            classifier_feats = self.dropout_prantik(classifier_feats)
            classifier_feats = self.classifier_prantik_2(classifier_feats)#torch.Size([2, 40401, 19])
            return classifier_feats,out
        return out

    def _decode(self, c1, c4, val_feat=False):
        #below print only when we need_fp=True
        #print(c1.shape, c4.shape)#torch.Size([8, 256, 201, 201]) torch.Size([8, 2048, 51, 51])
        c4 = self.head(c4)
        c4 = F.interpolate(c4, size=c1.shape[-2:], mode="bilinear", align_corners=True)

        c1 = self.reduce(c1)
        #print("check decode1", c1.shape, c4.shape)#torch.Size([8, 48, 201, 201]) torch.Size([8, 256, 201, 201])

        feature = torch.cat([c1, c4], dim=1)
        #print("check decode2", feature.shape)#torch.Size([8, 304, 201, 201])
        feature = self.fuse(feature)
        #print("check decode3", feature.shape)

        out = self.classifier(feature)
        #print("check decode4", out.shape)
        #print(dsds)
        if(val_feat):
            return feature,out
        return out


def ASPPConv(in_channels, out_channels, atrous_rate):
    block = nn.Sequential(nn.Conv2d(in_channels, out_channels, 3, padding=atrous_rate,
                                    dilation=atrous_rate, bias=False),
                          nn.BatchNorm2d(out_channels),
                          nn.ReLU(True))
    return block


class ASPPPooling(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ASPPPooling, self).__init__()
        self.gap = nn.Sequential(nn.AdaptiveAvgPool2d(1),
                                 nn.Conv2d(in_channels, out_channels, 1, bias=False),
                                 nn.BatchNorm2d(out_channels),
                                 nn.ReLU(True))

    def forward(self, x):
        h, w = x.shape[-2:]
        pool = self.gap(x)
        return F.interpolate(pool, (h, w), mode="bilinear", align_corners=True)


class ASPPModule(nn.Module):
    def __init__(self, in_channels, atrous_rates):
        super(ASPPModule, self).__init__()
        out_channels = in_channels // 8
        rate1, rate2, rate3 = atrous_rates

        self.b0 = nn.Sequential(nn.Conv2d(in_channels, out_channels, 1, bias=False),
                                nn.BatchNorm2d(out_channels),
                                nn.ReLU(True))
        self.b1 = ASPPConv(in_channels, out_channels, rate1)
        self.b2 = ASPPConv(in_channels, out_channels, rate2)
        self.b3 = ASPPConv(in_channels, out_channels, rate3)
        self.b4 = ASPPPooling(in_channels, out_channels)

        self.project = nn.Sequential(nn.Conv2d(5 * out_channels, out_channels, 1, bias=False),
                                     nn.BatchNorm2d(out_channels),
                                     nn.ReLU(True))

    def forward(self, x):
        feat0 = self.b0(x)
        feat1 = self.b1(x)
        feat2 = self.b2(x)
        feat3 = self.b3(x)
        feat4 = self.b4(x)
        y = torch.cat((feat0, feat1, feat2, feat3, feat4), 1)
        return self.project(y)
