# encoding: utf-8

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from functools import partial
from collections import OrderedDict

from resnet import get_resnet50


class SimpleRB(nn.Module):
    def __init__(self, in_channel, norm_layer, bn_momentum):
        super(SimpleRB, self).__init__()
        self.path = nn.Sequential(
            nn.Conv3d(in_channel, in_channel, kernel_size=3, padding=1, bias=False),
            norm_layer(in_channel, momentum=bn_momentum),
            nn.ReLU(),
            nn.Conv3d(in_channel, in_channel, kernel_size=3, padding=1, bias=False),
            norm_layer(in_channel, momentum=bn_momentum),
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        residual = x
        conv_path = self.path(x)
        out = residual + conv_path
        out = self.relu(out)
        return out


'''
3D Residual Block，3x3x3 conv ==> 3 smaller 3D conv, refered from DDRNet
'''
class Bottleneck3D(nn.Module):

    def __init__(self, inplanes, planes, norm_layer, stride=1, dilation=[1, 1, 1], expansion=4, downsample=None,
                 fist_dilation=1, multi_grid=1,
                 bn_momentum=0.0003):
        super(Bottleneck3D, self).__init__()
        # often，planes = inplanes // 4
        self.expansion = expansion
        self.conv1 = nn.Conv3d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = norm_layer(planes, momentum=bn_momentum)
        self.conv2 = nn.Conv3d(planes, planes, kernel_size=(1, 1, 3), stride=(1, 1, stride),
                               dilation=(1, 1, dilation[0]), padding=(0, 0, dilation[0]), bias=False)
        self.bn2 = norm_layer(planes, momentum=bn_momentum)
        self.conv3 = nn.Conv3d(planes, planes, kernel_size=(1, 3, 1), stride=(1, stride, 1),
                               dilation=(1, dilation[1], 1), padding=(0, dilation[1], 0), bias=False)
        self.bn3 = norm_layer(planes, momentum=bn_momentum)
        self.conv4 = nn.Conv3d(planes, planes, kernel_size=(3, 1, 1), stride=(stride, 1, 1),
                               dilation=(dilation[2], 1, 1), padding=(dilation[2], 0, 0), bias=False)
        self.bn4 = norm_layer(planes, momentum=bn_momentum)
        self.conv5 = nn.Conv3d(planes, planes * self.expansion, kernel_size=(1, 1, 1), bias=False)
        self.bn5 = norm_layer(planes * self.expansion, momentum=bn_momentum)

        self.relu = nn.ReLU(inplace=False)
        self.relu_inplace = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.dilation = dilation
        self.stride = stride

        self.downsample2 = nn.Sequential(
            nn.AvgPool3d(kernel_size=(1, stride, 1), stride=(1, stride, 1)),
            nn.Conv3d(planes, planes, kernel_size=1, stride=1, bias=False),
            norm_layer(planes, momentum=bn_momentum),
        )
        self.downsample3 = nn.Sequential(
            nn.AvgPool3d(kernel_size=(stride, 1, 1), stride=(stride, 1, 1)),
            nn.Conv3d(planes, planes, kernel_size=1, stride=1, bias=False),
            norm_layer(planes, momentum=bn_momentum),
        )
        self.downsample4 = nn.Sequential(
            nn.AvgPool3d(kernel_size=(stride, 1, 1), stride=(stride, 1, 1)),
            nn.Conv3d(planes, planes, kernel_size=1, stride=1, bias=False),
            norm_layer(planes, momentum=bn_momentum),
        )

    def forward(self, x):
        residual = x

        out1 = self.relu(self.bn1(self.conv1(x)))
        out2 = self.bn2(self.conv2(out1))
        out2_relu = self.relu(out2)

        out3 = self.bn3(self.conv3(out2_relu))
        if self.stride != 1:
            out2 = self.downsample2(out2)
        out3 = out3 + out2
        out3_relu = self.relu(out3)

        out4 = self.bn4(self.conv4(out3_relu))
        if self.stride != 1:
            out2 = self.downsample3(out2)
            out3 = self.downsample4(out3)
        out4 = out4 + out2 + out3

        out4_relu = self.relu(out4)
        out5 = self.bn5(self.conv5(out4_relu))

        if self.downsample is not None:
            residual = self.downsample(x)

        out = out5 + residual
        out_relu = self.relu(out)

        return out_relu
class attn_block(nn.Module):
    def __init__(self,features):
        super(attn_block, self).__init__()
        self.business_layer = []
        self.channel_avg_pool = nn.AdaptiveAvgPool3d(1)
        self.channel_max_pool = nn.AdaptiveMaxPool3d(1)
        self.channel_conv = nn.ModuleList([
            nn.Conv3d(2*features, features, kernel_size=1, stride=1, padding=0, bias=False),
            ]
        )
        self.channel_activation = nn.Sigmoid()
        self.spatial_conv_1 = nn.Conv3d(2,2,kernel_size=1)
        self.spatial_conv_2 = nn.Conv3d(2,1,kernel_size=3,padding=1)
        self.spatial_activation = nn.Sigmoid()
        self.business_layer.append(self.channel_avg_pool)
        self.business_layer.append(self.channel_max_pool)
        self.business_layer.append(self.channel_conv)
        self.business_layer.append(self.spatial_conv_1)
        self.business_layer.append(self.spatial_conv_2)
        # self.business_layer.append(self.spatial_activation)

    def forward(self,data):
        channel_max_descriptor = self.channel_max_pool(data)
        channel_avg_descriptor = self.channel_avg_pool(data)
        channel_descriptor = torch.cat([channel_max_descriptor,channel_avg_descriptor],dim=1)
        channel_descriptor = self.channel_conv[0](channel_descriptor)
        channel_descriptor = self.channel_activation(channel_descriptor)
        channel_refined_data = data.mul(channel_descriptor)

        spatial_max_descriptor,_ = torch.max(channel_refined_data,dim=1,keepdim=True)
        spatial_avg_descriptor = torch.mean(channel_refined_data,dim=1,keepdim=True)
        spatial_descriptor = torch.cat([spatial_max_descriptor,spatial_avg_descriptor],dim=1)
        spatial_descriptor = self.spatial_conv_1(spatial_descriptor)
        spatial_descriptor = self.spatial_conv_2(spatial_descriptor)
        spatial_descriptor = self.spatial_activation(spatial_descriptor)
        spatial_refined_data = channel_refined_data.mul(spatial_descriptor)



        return spatial_refined_data
class STAGE2(nn.Module):
    def __init__(self, class_num, norm_layer, resnet_out=2048, feature=512, ThreeDinit=True,
                 bn_momentum=0.1, pretrained_model=None, eval=False, freeze_bn=False):
        super(STAGE2, self).__init__()
        self.business_layer = []
        self.resnet_out = resnet_out
        self.feature = feature
        self.ThreeDinit = ThreeDinit
        self.pooling = nn.AvgPool3d(kernel_size=3, padding=1, stride=1)
        self.semantic_layer1_rgb = nn.Sequential(
            Bottleneck3D(feature, feature // 4, bn_momentum=bn_momentum, expansion=4, stride=2, downsample=
            nn.Sequential(
                nn.AvgPool3d(kernel_size=2, stride=2),
                nn.Conv3d(feature, feature,
                          kernel_size=1, stride=1, bias=False),
                norm_layer(feature, momentum=bn_momentum),
            ), norm_layer=norm_layer),
            Bottleneck3D(feature, feature // 4, bn_momentum=bn_momentum, norm_layer=norm_layer, dilation=[1, 1, 1]),
            Bottleneck3D(feature, feature // 4, bn_momentum=bn_momentum, norm_layer=norm_layer, dilation=[2, 2, 2]),
            Bottleneck3D(feature, feature // 4, bn_momentum=bn_momentum, norm_layer=norm_layer, dilation=[3, 3, 3]),
        )
        self.business_layer.append(self.semantic_layer1_rgb)
        self.semantic_layer2_rgb = nn.Sequential(
            Bottleneck3D(feature, feature // 4, bn_momentum=bn_momentum, expansion=8, stride=2, downsample=
            nn.Sequential(
                nn.AvgPool3d(kernel_size=2, stride=2),
                nn.Conv3d(feature, feature * 2,
                          kernel_size=1, stride=1, bias=False),
                norm_layer(feature * 2, momentum=bn_momentum),
            ), norm_layer=norm_layer),
            Bottleneck3D(feature * 2, feature // 2, bn_momentum=bn_momentum, norm_layer=norm_layer, dilation=[1, 1, 1]),
            Bottleneck3D(feature * 2, feature // 2, bn_momentum=bn_momentum, norm_layer=norm_layer, dilation=[2, 2, 2]),
            Bottleneck3D(feature * 2, feature // 2, bn_momentum=bn_momentum, norm_layer=norm_layer, dilation=[3, 3, 3]),
        )
        self.business_layer.append(self.semantic_layer2_rgb)

        self.semantic_layer1_tsdf = nn.Sequential(
            Bottleneck3D(feature, feature // 4, bn_momentum=bn_momentum, expansion=4, stride=2, downsample=
            nn.Sequential(
                nn.AvgPool3d(kernel_size=2, stride=2),
                nn.Conv3d(feature, feature,
                          kernel_size=1, stride=1, bias=False),
                norm_layer(feature, momentum=bn_momentum),
            ), norm_layer=norm_layer),
            Bottleneck3D(feature, feature // 4, bn_momentum=bn_momentum, norm_layer=norm_layer, dilation=[1, 1, 1]),
            Bottleneck3D(feature, feature // 4, bn_momentum=bn_momentum, norm_layer=norm_layer, dilation=[2, 2, 2]),
            Bottleneck3D(feature, feature // 4, bn_momentum=bn_momentum, norm_layer=norm_layer, dilation=[3, 3, 3]),
        )
        self.business_layer.append(self.semantic_layer1_tsdf)
        self.semantic_layer2_tsdf = nn.Sequential(
            Bottleneck3D(feature, feature // 4, bn_momentum=bn_momentum, expansion=8, stride=2, downsample=
            nn.Sequential(
                nn.AvgPool3d(kernel_size=2, stride=2),
                nn.Conv3d(feature, feature * 2,
                          kernel_size=1, stride=1, bias=False),
                norm_layer(feature * 2, momentum=bn_momentum),
            ), norm_layer=norm_layer),
            Bottleneck3D(feature * 2, feature // 2, bn_momentum=bn_momentum, norm_layer=norm_layer, dilation=[1, 1, 1]),
            Bottleneck3D(feature * 2, feature // 2, bn_momentum=bn_momentum, norm_layer=norm_layer, dilation=[2, 2, 2]),
            Bottleneck3D(feature * 2, feature // 2, bn_momentum=bn_momentum, norm_layer=norm_layer, dilation=[3, 3, 3]),
        )
        self.business_layer.append(self.semantic_layer2_tsdf)

        self.classify_semantic = nn.ModuleList([
            nn.Sequential(
                nn.ConvTranspose3d(feature * 2, feature, kernel_size=3, stride=2, padding=1, dilation=1,
                                   output_padding=1),
                norm_layer(feature, momentum=bn_momentum),
                nn.ReLU(inplace=False),
            ),
            nn.Sequential(
                nn.ConvTranspose3d(feature, feature, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1),
                norm_layer(feature, momentum=bn_momentum),
                nn.ReLU(inplace=False),
            ),
            nn.Sequential(
                nn.Dropout3d(.1),
                nn.Conv3d(feature, class_num, kernel_size=1, bias=True)
            ),]
        )
        self.business_layer.append(self.classify_semantic)


        self.conv1 = nn.Sequential(
            nn.Conv3d(feature, feature, kernel_size=3, padding=1, bias=False),
            norm_layer(feature, momentum=bn_momentum),
            nn.ReLU(inplace=False),
        )
        self.conv0 = nn.Sequential(
            nn.Conv3d(feature, feature, kernel_size=3, padding=1, bias=False),
            norm_layer(feature, momentum=bn_momentum),
            nn.ReLU(inplace=False),
        )
        self.final_conv = nn.Sequential(
            nn.Conv3d(feature, feature, kernel_size=3, padding=1, bias=False),
            norm_layer(feature, momentum=bn_momentum),
            nn.ReLU(inplace=False),
        )
        self.business_layer.append(self.conv1)
        self.business_layer.append(self.conv0)
        self.business_layer.append(self.final_conv)

        self.oper_tsdf = nn.Sequential(
            nn.Conv3d(1, 3, kernel_size=3, padding=1, bias=False),
            norm_layer(3, momentum=bn_momentum),
            nn.ReLU(),
            nn.Conv3d(3, 64, kernel_size=3, padding=1, bias=False),
            norm_layer(64, momentum=bn_momentum),
            nn.ReLU(),
            nn.Conv3d(64, feature, kernel_size=3, padding=1, bias=False),
            norm_layer(feature, momentum=bn_momentum),
            nn.ReLU(inplace=False),
            Bottleneck3D(feature, feature // 4, bn_momentum=bn_momentum, norm_layer=norm_layer, dilation=[1, 1, 1]),
            Bottleneck3D(feature, feature // 4, bn_momentum=bn_momentum, norm_layer=norm_layer, dilation=[2, 2, 2]),
            Bottleneck3D(feature, feature // 4, bn_momentum=bn_momentum, norm_layer=norm_layer, dilation=[3, 3, 3]),
            # Bottleneck3D(feature, feature // 4, bn_momentum=bn_momentum, norm_layer=norm_layer, dilation=[1, 1, 1]),


        )

        self.business_layer.append(self.oper_tsdf)


    def forward(self, raw, tsdf):
        # raw = torch.zeros_like(raw)
        semantic_rgb_0 = raw
        semantic_tsdf_0 = self.oper_tsdf(tsdf)
        seg_fea_0 = semantic_rgb_0+semantic_tsdf_0


        semantic_rgb_1 = self.semantic_layer1_rgb(semantic_rgb_0)
        semantic_tsdf_1 = self.semantic_layer1_tsdf(semantic_tsdf_0)
        seg_fea_1 = semantic_rgb_1+semantic_tsdf_1

        semantic_rgb_2 = self.semantic_layer2_rgb(semantic_rgb_1)
        semantic_tsdf_2 = self.semantic_layer2_tsdf(semantic_tsdf_1)
        seg_fea_2 = semantic_rgb_2+semantic_tsdf_2

        up_sem1 = self.classify_semantic[0](seg_fea_2) + self.conv1(seg_fea_1)
        up_sem2 = self.classify_semantic[1](up_sem1) + self.conv0(seg_fea_0)
        pred_semantic2 = self.final_conv(up_sem2)
        pred_semantic = self.classify_semantic[2](pred_semantic2)
        results = {'pred_semantic': pred_semantic,
                   'seg_feature_0':seg_fea_0,'seg_feature_1':seg_fea_1,'seg_feature_2':seg_fea_2,
                   'up_sem1':up_sem1,'up_sem2':up_sem2, 'final_seg_feature':pred_semantic2,
                   'semantic_rgb_0':semantic_rgb_0,'semantic_tsdf_0':semantic_tsdf_0,
                   'semantic_rgb_1':semantic_rgb_1,'semantic_tsdf_1':semantic_tsdf_1,
                   'semantic_rgb_2':semantic_rgb_2,'semantic_tsdf_2':semantic_tsdf_2,
                           }

        return results

class STAGE3(nn.Module):
    def __init__(self, class_num, norm_layer, resnet_out=2048, feature=512, ThreeDinit=True,
                 bn_momentum=0.1, pretrained_model=None, eval=False, freeze_bn=False):
        super(STAGE3, self).__init__()
        self.business_layer = []
        self.resnet_out = resnet_out
        self.feature = feature
        self.ThreeDinit = ThreeDinit
        self.pooling = nn.AvgPool3d(kernel_size=3, padding=1, stride=1)
        self.semantic_layer1_rgb = nn.Sequential(
            Bottleneck3D(feature, feature // 4, bn_momentum=bn_momentum, expansion=4, stride=2, downsample=
            nn.Sequential(
                nn.AvgPool3d(kernel_size=2, stride=2),
                nn.Conv3d(feature, feature,
                          kernel_size=1, stride=1, bias=False),
                norm_layer(feature, momentum=bn_momentum),
            ), norm_layer=norm_layer),
            Bottleneck3D(feature, feature // 4, bn_momentum=bn_momentum, norm_layer=norm_layer, dilation=[1, 1, 1]),
            Bottleneck3D(feature, feature // 4, bn_momentum=bn_momentum, norm_layer=norm_layer, dilation=[2, 2, 2]),
            Bottleneck3D(feature, feature // 4, bn_momentum=bn_momentum, norm_layer=norm_layer, dilation=[3, 3, 3]),
        )
        self.business_layer.append(self.semantic_layer1_rgb)
        self.semantic_layer2_rgb = nn.Sequential(
            Bottleneck3D(feature, feature // 4, bn_momentum=bn_momentum, expansion=8, stride=2, downsample=
            nn.Sequential(
                nn.AvgPool3d(kernel_size=2, stride=2),
                nn.Conv3d(feature, feature * 2,
                          kernel_size=1, stride=1, bias=False),
                norm_layer(feature * 2, momentum=bn_momentum),
            ), norm_layer=norm_layer),
            Bottleneck3D(feature * 2, feature // 2, bn_momentum=bn_momentum, norm_layer=norm_layer, dilation=[1, 1, 1]),
            Bottleneck3D(feature * 2, feature // 2, bn_momentum=bn_momentum, norm_layer=norm_layer, dilation=[2, 2, 2]),
            Bottleneck3D(feature * 2, feature // 2, bn_momentum=bn_momentum, norm_layer=norm_layer, dilation=[3, 3, 3]),
        )
        self.business_layer.append(self.semantic_layer2_rgb)


        self.classify_semantic = nn.ModuleList([
            nn.Sequential(
                nn.ConvTranspose3d(feature * 2, feature, kernel_size=3, stride=2, padding=1, dilation=1,
                                   output_padding=1),
                norm_layer(feature, momentum=bn_momentum),
                nn.ReLU(inplace=False),
            ),
            nn.Sequential(
                nn.ConvTranspose3d(feature, feature, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1),
                norm_layer(feature, momentum=bn_momentum),
                nn.ReLU(inplace=False),
            ),
            nn.Sequential(
                nn.Dropout3d(.1),
                nn.Conv3d(feature, class_num, kernel_size=1, bias=True)
            ),]
        )
        self.business_layer.append(self.classify_semantic)


        self.conv1 = nn.Sequential(
            nn.Conv3d(feature, feature, kernel_size=3, padding=1, bias=False),
            norm_layer(feature, momentum=bn_momentum),
            nn.ReLU(inplace=False),
        )
        self.conv0 = nn.Sequential(
            nn.Conv3d(feature, feature, kernel_size=3, padding=1, bias=False),
            norm_layer(feature, momentum=bn_momentum),
            nn.ReLU(inplace=False),
        )
        self.final_conv = nn.Sequential(
            nn.Conv3d(feature, feature, kernel_size=3, padding=1, bias=False),
            norm_layer(feature, momentum=bn_momentum),
            nn.ReLU(inplace=False),
        )
        self.business_layer.append(self.conv1)
        self.business_layer.append(self.conv0)
        self.business_layer.append(self.final_conv)





    def knn_init(self,raw,tsdf,pred_semantic,depth_mapping_3d,gt):
        bs, c, d, h, w = raw.size()
        depth_mapping_3d = depth_mapping_3d.reshape(bs,d,h,w)
        raw = raw.permute(0,2,3,4,1).clone().detach()
        each_raws = []
        tsdf = tsdf.reshape(bs,d,h,w)
        pred_semantic = torch.argmax(pred_semantic,dim=1)
        for b in range(bs):
            each_tsdf = tsdf[b]
            each_raw = raw[b]
            each_pred = pred_semantic[b]
            each_depth_mapping_3d = depth_mapping_3d[b]
            for i in range(1,12):
                surface_class_region = (each_pred==i) & (each_depth_mapping_3d<307200) & (each_tsdf>=-1)
                back_class_region = (each_pred==i) & (each_depth_mapping_3d==307200) & (each_tsdf>=-1) & (each_tsdf<=0)
                if surface_class_region.sum().item()>0 and back_class_region.sum().item()>0:
                    each_raw[back_class_region] = each_raw[surface_class_region].mean(dim=0)
            each_raws.append(each_raw)
        raw = torch.stack(each_raws,dim=0)
        raw = raw.permute(0,4,1,2,3)
        return raw

    def forward(self, raw, tsdf, pred_semantic, depth_mapping_3d,gt,early_results):
        raw = self.knn_init(raw,tsdf,pred_semantic,depth_mapping_3d,gt)
        semantic_rgb_0 = raw
        semantic_tsdf_0 = early_results['semantic_tsdf_0']
        seg_fea_0 = semantic_rgb_0+semantic_tsdf_0


        semantic_rgb_1 = self.semantic_layer1_rgb(semantic_rgb_0)
        semantic_tsdf_1 =  early_results['semantic_tsdf_1']
        seg_fea_1 = semantic_rgb_1+semantic_tsdf_1

        semantic_rgb_2 = self.semantic_layer2_rgb(semantic_rgb_1)
        semantic_tsdf_2 =  early_results['semantic_tsdf_2']
        seg_fea_2 = semantic_rgb_2+semantic_tsdf_2

        up_sem1 = self.classify_semantic[0](seg_fea_2) + self.conv1(seg_fea_1)
        up_sem2 = self.classify_semantic[1](up_sem1) + self.conv0(seg_fea_0)
        pred_semantic2 = self.final_conv(up_sem2)
        pred_semantic = self.classify_semantic[2](pred_semantic2)
        results = {
            'pred_semantic_refine': pred_semantic,
            }

        return results
class Bottleneck(nn.Module):
    expansion = 4
    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None, fist_dilation=1, multi_grid=1, bn_momentum=0.0003, BatchNorm2d=nn.BatchNorm2d):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = BatchNorm2d(planes, momentum=bn_momentum)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=dilation*multi_grid, dilation=dilation*multi_grid, bias=False)
        self.bn2 = BatchNorm2d(planes, momentum=bn_momentum)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = BatchNorm2d(planes * 4, momentum=bn_momentum)
        self.relu = nn.ReLU(inplace=False)
        self.relu_inplace = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.dilation = dilation
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out = out + residual
        out = self.relu_inplace(out)
        return out

class Projection(nn.Module):
    def __init__(self, class_num, norm_layer, resnet_out=2048, feature=512, ThreeDinit=True,
                 bn_momentum=0.1, pretrained_model=None, eval=False, freeze_bn=False):
        super(Projection, self).__init__()
        self.business_layer = []

        if eval:
            self.downsample = nn.Sequential(
                nn.Conv2d(resnet_out, feature, kernel_size=1, bias=False),
                nn.BatchNorm2d(feature, momentum=bn_momentum),
                nn.ReLU()
            )
        else:
            self.downsample = nn.Sequential(
                nn.Conv2d(resnet_out, feature, kernel_size=1, bias=False),
                norm_layer(feature, momentum=bn_momentum),
                nn.ReLU()
            )
        self.business_layer.append(self.downsample)

        self.resnet_out = resnet_out
        self.feature = feature

        self.conv_rgb = Bottleneck(feature,feature//4)
        self.business_layer.append(self.conv_rgb)


    def forward(self, feature2d, depth_mapping_3d):
        # reduce the channel of 2D feature map
        if self.resnet_out != self.feature:
            feature2d = self.downsample(feature2d)
        feature2d = self.conv_rgb(feature2d)
        feature2d = F.interpolate(feature2d, scale_factor=16, mode='bilinear', align_corners=True)

        '''
        project 2D feature to 3D space
        '''
        b, c, h, w = feature2d.shape
        feature2d = feature2d.view(b, c, h * w).permute(0, 2, 1)  # b x h*w x c

        zerosVec = torch.zeros(b, 1, c).cuda()  # for voxels that could not be projected from the depth map, we assign them zero vector
        segVec = torch.cat((feature2d, zerosVec), 1)

        segres = [torch.index_select(segVec[i], 0, depth_mapping_3d[i]) for i in range(b)]
        segres = torch.stack(segres).permute(0, 2, 1).contiguous().view(b, c, 60, 36, 60)  # B, (channel), 60, 36, 60

        return segres


'''
main network2d
'''
class baseline_resnet50_init_reuse(nn.Module):
    def __init__(self, class_num, norm_layer, resnet_out=2048, feature=512, ThreeDinit=True,
                 bn_momentum=0.1, pretrained_model=None, eval=False, freeze_bn=False):
        super(baseline_resnet50_init_reuse, self).__init__()
        self.business_layer = []

        if eval:
            self.backbone = get_resnet50(num_classes=19, dilation=[1, 1, 1, 2], bn_momentum=bn_momentum,
                                         is_fpn=False,
                                         BatchNorm2d=nn.BatchNorm2d)
        else:
            self.backbone = get_resnet50(num_classes=19, dilation=[1, 1, 1, 2], bn_momentum=bn_momentum,
                                         is_fpn=False,
                                         BatchNorm2d=norm_layer)
        self.dilate = 2
        for m in self.backbone.layer4.children():
            m.apply(partial(self._nostride_dilate, dilate=self.dilate))
            self.dilate *= 2



        self.proj = Projection(class_num, norm_layer, resnet_out=resnet_out, feature=feature, ThreeDinit=ThreeDinit,
                             bn_momentum=bn_momentum, pretrained_model=pretrained_model, eval=eval, freeze_bn=freeze_bn)
        self.business_layer += self.proj.business_layer

        self.early_fusion = STAGE2(class_num, norm_layer, resnet_out=resnet_out, feature=feature, ThreeDinit=ThreeDinit,
                                   bn_momentum=bn_momentum, pretrained_model=pretrained_model, eval=eval, freeze_bn=freeze_bn)
        self.business_layer += self.early_fusion.business_layer

        self.late_fusion = STAGE3(class_num, norm_layer, resnet_out=resnet_out, feature=feature, ThreeDinit=ThreeDinit,
                                   bn_momentum=bn_momentum, pretrained_model=pretrained_model, eval=eval, freeze_bn=freeze_bn)
        self.business_layer += self.late_fusion.business_layer



    def forward(self,img, depth_mapping_3d, tsdf, sketch_gt,seg_2d,gt):

        h, w = img.size(2), img.size(3)

        feature2d = self.backbone(img) #C=12
        feature3d = self.proj(feature2d, depth_mapping_3d) #B,C,60,36,60

        early_results = self.early_fusion(feature3d, tsdf)
        late_results = self.late_fusion(feature3d, tsdf, early_results['pred_semantic'],depth_mapping_3d,gt,early_results)
        results = {'pred_semantic':early_results['pred_semantic'],
                   'pred_semantic_refine':late_results['pred_semantic_refine']}
        return results

    # @staticmethod
    def _nostride_dilate(self, m, dilate):
        if isinstance(m, nn.Conv2d):
            if m.stride == (2, 2):
                m.stride = (1, 1)
                if m.kernel_size == (3, 3):
                    m.dilation = (dilate, dilate)
                    m.padding = (dilate, dilate)

            else:
                if m.kernel_size == (3, 3):
                    m.dilation = (dilate, dilate)
                    m.padding = (dilate, dilate)


if __name__ == '__main__':
    model = Network(class_num=12, norm_layer=nn.BatchNorm3d, feature=128, eval=True)
    # print(model)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval() ######## this line change the self.training property

    left = torch.rand(1, 3, 480, 640).cuda()
    right = torch.rand(1, 3, 480, 640).cuda()
    depth_mapping_3d = torch.from_numpy(np.ones((1, 129600)).astype(np.int64)).long().cuda()
    tsdf = torch.rand(1, 1, 60, 36, 60).cuda()

    out = model(left, depth_mapping_3d, tsdf, None)
