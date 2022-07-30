# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
import math
from config import cfg
# feat_dims:list[dim1,dim2,dim3..] 构建全连接层
def make_linear_layers(feat_dims, relu_final=True):
    layers = []
    # 减1是为了防止维度列表越界
    for i in range(len(feat_dims)-1):
        layers.append(nn.Linear(feat_dims[i], feat_dims[i+1]))

        # Do not use ReLU for final estimation
        if i < len(feat_dims)-2 or (i == len(feat_dims)-2 and relu_final):
            layers.append(nn.ReLU(inplace=True))

    return nn.Sequential(*layers)
# 构建conv
def make_conv_layers(feat_dims, kernel=3, stride=1, padding=1, bnrelu_final=True):
    layers = []
    for i in range(len(feat_dims)-1):
        layers.append(
            #out=(input+2*padding-kernel_size)/stride+1
            nn.Conv2d(
                in_channels=feat_dims[i],
                out_channels=feat_dims[i+1],
                kernel_size=kernel,
                stride=stride,
                padding=padding
                ))
        # Do not use BN and ReLU for final estimation
        if i < len(feat_dims)-2 or (i == len(feat_dims)-2 and bnrelu_final):
            layers.append(nn.BatchNorm2d(feat_dims[i+1]))
            layers.append(nn.ReLU(inplace=True))

    return nn.Sequential(*layers)
# 构建反卷积层
def make_deconv_layers(feat_dims, bnrelu_final=True):
    layers = []
    for i in range(len(feat_dims)-1):
        layers.append(
            #out=(input-1)*stride+outputpadding-2*padding+kernel_size
            nn.ConvTranspose2d(
                in_channels=feat_dims[i],
                out_channels=feat_dims[i+1],
                kernel_size=4,
                stride=2,
                padding=1,
                output_padding=0,
                bias=False))

        # Do not use BN and ReLU for final estimation
        if i < len(feat_dims)-2 or (i == len(feat_dims)-2 and bnrelu_final):
            layers.append(nn.BatchNorm2d(feat_dims[i+1]))
            layers.append(nn.ReLU(inplace=True))

    return nn.Sequential(*layers)

class Interpolate(nn.Module):
    # scale_factor (float or Tuple[float]): spatial 尺寸的缩放因子.
    # mode上采样算法:nearest, linear, bilinear, trilinear, area.默认为nearest.
    def __init__(self, scale_factor, mode):
        super(Interpolate, self).__init__()
        #支持3D(N,C,W) 4D(N,C,H,W) 5D(N,C,D,H,W)
        self.interp = F.interpolate
        self.scale_factor = scale_factor
        self.mode = mode
        
    def forward(self, x):
        """
         align_corners(bool, optional)如果align_corners = True,则对齐input和output的角点像素(cornerpixels)保持在角点像素的值.
        只会对mode = linear, bilinear和trilinear有作用.默认是False.
        """
        x = self.interp(x, scale_factor=self.scale_factor, mode=self.mode, align_corners=False)
        return x

def make_upsample_layers(feat_dims, bnrelu_final=True):
    layers = []
    for i in range(len(feat_dims)-1):
        layers.append(
            Interpolate(2, 'bilinear'))
        layers.append(
            nn.Conv2d(
                in_channels=feat_dims[i],
                out_channels=feat_dims[i+1],
                kernel_size=3,
                stride=1,
                padding=1
                ))

        # Do not use BN and ReLU for final estimation
        if i < len(feat_dims)-2 or (i == len(feat_dims)-2 and bnrelu_final):
            layers.append(nn.BatchNorm2d(feat_dims[i+1]))
            layers.append(nn.ReLU(inplace=True))

    return nn.Sequential(*layers)

class ResBlock(nn.Module):
    def __init__(self, in_feat, out_feat):
        super(ResBlock, self).__init__()
        self.in_feat = in_feat
        self.out_feat = out_feat

        self.conv = make_conv_layers([in_feat, out_feat, out_feat], bnrelu_final=False)
        self.bn = nn.BatchNorm2d(out_feat)
        if self.in_feat != self.out_feat:
            self.shortcut_conv = nn.Conv2d(in_feat,out_feat,kernel_size=1,stride=1,padding=0)
            self.shortcut_bn = nn.BatchNorm2d(out_feat)

    def forward(self, input):
        x = self.bn(self.conv(input))
        if self.in_feat != self.out_feat:
            x = F.relu(x + self.shortcut_bn(self.shortcut_conv(input)))
        else:
            x = F.relu(x + input)
        return x

# Conv2d的规定输入数据格式为(batch, channel, Height, Width)
# Conv3d的规定输入数据格式为(batch, channel, Depth, Height, Width)
def make_conv3d_layers(feat_dims, kernel=3, stride=1, padding=1, bnrelu_final=True):
    layers = []
    for i in range(len(feat_dims)-1):
        layers.append(
            nn.Conv3d(
                in_channels=feat_dims[i],
                out_channels=feat_dims[i+1],
                kernel_size=kernel,
                stride=stride,
                padding=padding
                ))
        # Do not use BN and ReLU for final estimation
        if i < len(feat_dims)-2 or (i == len(feat_dims)-2 and bnrelu_final):
            layers.append(nn.BatchNorm3d(feat_dims[i+1]))
            layers.append(nn.ReLU(inplace=True))

    return nn.Sequential(*layers)

def make_deconv3d_layers(feat_dims, bnrelu_final=True):
    layers = []
    for i in range(len(feat_dims)-1):
        layers.append(
            nn.ConvTranspose3d(
                in_channels=feat_dims[i],
                out_channels=feat_dims[i+1],
                kernel_size=4,
                stride=2,
                padding=1,
                output_padding=0,
                bias=False))

        # Do not use BN and ReLU for final estimation
        if i < len(feat_dims)-2 or (i == len(feat_dims)-2 and bnrelu_final):
            layers.append(nn.BatchNorm3d(feat_dims[i+1]))
            layers.append(nn.ReLU(inplace=True))

    return nn.Sequential(*layers)

