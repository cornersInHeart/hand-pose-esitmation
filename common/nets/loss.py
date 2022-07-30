# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
from config import cfg
import math

# 手姿态损失
class JointHeatmapLoss(nn.Module):
    def __ini__(self):
        super(JointHeatmapLoss, self).__init__()

    def forward(self, joint_out, joint_gt, joint_valid):
        # joint_valid.size():[42,1] 当只有单手时 则将损失置为0
        loss = (joint_out - joint_gt)**2 * joint_valid[:,:,None,None,None]
        return loss

# 手类型loss
class HandTypeLoss(nn.Module):
    def __init__(self):
        super(HandTypeLoss, self).__init__()

    def forward(self, hand_type_out, hand_type_gt, hand_type_valid):
        # F.cross_entropy()对应的是torch.nn.CrossEntropyLoss,使用自动添加logsoftmax再算loss（是nn.LogSoftmax和nn.NLLLoss的融合）
        # reduction -elementwise_mean(默认):对N个样本的loss进行求平均之后返回 ，sum：对n个样本的loss求和,none:返回n分样本的loss
        loss = F.binary_cross_entropy(hand_type_out, hand_type_gt, reduction='none')
        #在axis=1的维度上进行计算平均值
        loss = loss.mean(1)
        loss = loss * hand_type_valid

        return loss

# 相对深度loss
class RelRootDepthLoss(nn.Module):
    def __init__(self):
        super(RelRootDepthLoss, self).__init__()

    def forward(self, root_depth_out, root_depth_gt, root_valid):
        loss = torch.abs(root_depth_out - root_depth_gt) * root_valid
        return loss

