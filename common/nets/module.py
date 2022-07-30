# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.



import torch
import torch.nn as nn
from torch.nn import functional as F
from config import cfg
from nets.layer import make_linear_layers, make_conv_layers, make_deconv_layers, make_upsample_layers
from nets.resnet import ResNetBackbone
import numpy as np
from PIL import Image


# 主干网络，使用ResNet
class BackboneNet(nn.Module):
    def __init__(self):
        super(BackboneNet, self).__init__()
        self.resnet = ResNetBackbone(cfg.resnet_type)

    def init_weights(self):
        self.resnet.init_weights()

    def forward(self, img):
        # torch.size([4,2048,8,8])-[batch,channel,h,w]
        img_feat, x1, x2, x3 = self.resnet(img)
        # print("img_feat:",img_feat.shape)
        return img_feat, x1, x2, x3




class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1):
        # resnet50 moudle
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,padding=1)
        # self.conv2 = nn.Sequential(
        #         nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, groups=planes),
        #         nn.BatchNorm2d(planes),
        #         nn.Conv2d(planes, planes, kernel_size=1, stride=1)
        #     )
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 2, kernel_size=1)
        self.bn3 = nn.BatchNorm2d(planes * 2)
        self.relu = nn.ReLU(inplace=True)

        self.downsample = nn.Sequential(
            nn.Conv2d(inplanes, planes * 2,kernel_size=1, stride=stride),
            nn.BatchNorm2d(planes * 2),
        )

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

        out += residual
        out = self.relu(out)

        return out







class SElayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SElayer, self).__init__()
        # channel
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # add
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel),
            nn.Sigmoid()
        )


    def forward(self, x):
        # channel
        b, c, _, _ = x.size()
        out_avg = self.avg_pool(x).view(b, c)
        # add
        out_max = self.max_pool(x).view(b,c)
        out = out_avg+out_max

        out = self.fc(out).view(b, c, 1, 1)
        out_se = x * out.expand_as(x)
        output = out_se + x
        return output




# 检测手势位置信息
class PoseNet(nn.Module):
    def __init__(self, joint_num):
        super(PoseNet, self).__init__()
        self.channels = [2048, 1024, 512, 256]

        self.joint_num = joint_num  # single hand

        se_layer = []
        for i in range(len(self.channels)):
            se_layer.append(SElayer(self.channels[i]))
        self.se_layer = nn.ModuleList(se_layer)

        self.gloabalnet_1 = globalNet(self.channels, (64, 64), 17)

        # self.de_regression_1 = Deconv_regression(self.channels)
        # self.joint_deconv_1 = make_deconv_layers([2048,256,256,256])
        self.refine_net_1 = refineNet(256)
        # output_hm_shape = (64, 64, 64) # (depth, height, width)
        # self.joint_conv_1 = make_conv_layers([256,self.joint_num*cfg.output_hm_shape[0]],kernel=1,stride=1,padding=0,bnrelu_final=False)
        self.joint_conv_1 = make_conv_layers([1024, self.joint_num * cfg.output_hm_shape[0]], kernel=1, stride=1,
                                             padding=0, bnrelu_final=False)

        # self.gloabalnet_2 = globalNet(self.channels, (64, 64), 17)

        # self.de_regression_2 = Deconv_regression(self.channels)
        # self.joint_deconv_2 = make_deconv_layers([2048,256,256,256])
        self.refine_net_2 = refineNet(256)
        # self.joint_conv_2 = make_conv_layers([256,self.joint_num*cfg.output_hm_shape[0]],kernel=1,stride=1,padding=0,bnrelu_final=False)
        self.joint_conv_2 = make_conv_layers([1024, self.joint_num * cfg.output_hm_shape[0]], kernel=1, stride=1,
                                             padding=0, bnrelu_final=False)

        # output_root_hm_shape = 64  depth axis
        self.root_fc = make_linear_layers([2048, 512, cfg.output_root_hm_shape], relu_final=False)
        # 判断左右手是否存在
        self.hand_fc = make_linear_layers([2048, 512, 2], relu_final=False)

    def soft_argmax_1d(self, heatmap1d):
        heatmap1d = F.softmax(heatmap1d, 1)
        # 向多个gpu广播一个张量。
        # accu = heatmap1d * torch.cuda.comm.broadcast(torch.arange(cfg.output_root_hm_shape).type(torch.cuda.FloatTensor), devices=[heatmap1d.device.index])[0]
        accu = heatmap1d * torch.arange(cfg.output_root_hm_shape).float().cuda()[None, :]
        coord = accu.sum(dim=1)
        return coord

    def forward(self, img_feat):
        # (4,256,64,64)
        img_feat = list(img_feat)
        for i in range(len(img_feat)):
            img_feat[i] = self.se_layer[i](img_feat[i])
        gobalnet_feat_1 = self.gloabalnet_1(img_feat)
        # gobalnet_feat_2 = self.gloabalnet_1(img_feat)

        # (4,1024,64,64)

        # origin img_feat : torch.Size([1, 2048, 8, 8]) -img_feat[0]
        # left size:[4,256,64,64] out=(8-1)*2-2*1+4=16;(16-1)*2-2*1+4=32;(32-1)*2-2*1+4=64
        # joint_img_feat_1 = self.joint_deconv_1(img_feat[0])
        # joints_img_feat_1 = self.de_regression_1(img_feat)
        refinenet_1 = self.refine_net_1(gobalnet_feat_1)

        # size:[4,21,64,64,64]  通过卷积得h,w=64--得到每个关节点的输出坐标
        # joint_heatmap3d_1 = self.joint_conv_1(joints_img_feat_1+gobalnet_feat_1[3]) .view(-1,self.joint_num,cfg.output_hm_shape[0],cfg.output_hm_shape[1],cfg.output_hm_shape[2])
        joint_heatmap3d_1 = self.joint_conv_1(refinenet_1).view(-1, self.joint_num, cfg.output_hm_shape[0],
                                                                cfg.output_hm_shape[1], cfg.output_hm_shape[2])

        # right
        # joint_img_feat_2 = self.joint_deconv_2(img_feat[0])
        # joints_img_feat_2 = self.de_regression_2(img_feat)
        refinenet_2 = self.refine_net_2(gobalnet_feat_1)
        # joint_heatmap3d_2 = self.joint_conv_2(joints_img_feat_2+gobalnet_feat_2[3]).view(-1,self.joint_num,cfg.output_hm_shape[0],cfg.output_hm_shape[1],cfg.output_hm_shape[2])
        joint_heatmap3d_2 = self.joint_conv_2(refinenet_2).view(-1, self.joint_num, cfg.output_hm_shape[0],
                                                                cfg.output_hm_shape[1], cfg.output_hm_shape[2])
        # 左右手拼接 2.5D右手和左手姿态估计 [4,42,64,64,64]
        joint_heatmap3d = torch.cat((joint_heatmap3d_1, joint_heatmap3d_2), 1)
        # torch.cuda.empty_cache()

        # size:[4,2048] F.avg_pool2d(input,kernel_size)
        img_feat_gap = F.avg_pool2d(img_feat[0], (img_feat[0].shape[2], img_feat[0].shape[3])).view(-1, 2048)
        # size:[4,64] -1D heatmap
        root_heatmap1d = self.root_fc(img_feat_gap)
        # shape:[4,1] 右手相对左手深度估计
        root_depth = self.soft_argmax_1d(root_heatmap1d).view(-1, 1)
        # shape:[4,2] 预测左右手存在概率
        hand_type = torch.sigmoid(self.hand_fc(img_feat_gap))
        return joint_heatmap3d, root_depth, hand_type


# if __name__ == "__main__":
#     device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#     input = torch.randn(1,3,256,256).to(device)
#     resnet = BackboneNet().to(device)
#     img_feat = resnet(input)
#     print(len(img_feat))
#     globalnet = globalNet([2048, 1024, 512, 256],(64, 64),17).to(device)
#     refine_net = refineNet(256,(64, 64),17).to(device)
#     global_ms = globalnet(img_feat)
#     output = refine_net(global_ms)
#     print("global_fms:",global_ms[3].shape)
#     print(output.shape)
"""
x1: torch.Size([1, 256, 64, 64])
x2: torch.Size([1, 512, 32, 32])
x3: torch.Size([1, 1024, 16, 16])
x4: torch.Size([1, 2048, 8, 8])
"""