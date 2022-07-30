# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import torch
import torch.nn as nn
import torch.nn.functional as F
from nets.module import BackboneNet, PoseNet
from nets.loss import JointHeatmapLoss, HandTypeLoss, RelRootDepthLoss
from config import cfg
import math

class Model(nn.Module):
    def __init__(self,backbone_net, pose_net):
        super(Model, self).__init__()

        # modules
        self.backbone_net = backbone_net
        self.pose_net = pose_net
        # self.hourglass = hourglass
          
        # loss functions
        self.joint_heatmap_loss = JointHeatmapLoss()
        self.rel_root_depth_loss = RelRootDepthLoss()
        self.hand_type_loss = HandTypeLoss()
     
    def render_gaussian_heatmap(self, joint_coord):
        #output_hm_shape = (64, 64, 64)   (depth, height, width)
        x = torch.arange(cfg.output_hm_shape[2])
        y = torch.arange(cfg.output_hm_shape[1])
        z = torch.arange(cfg.output_hm_shape[0])
        # 生成网格，可以用于生成坐标  size:[64,64,64]
        zz,yy,xx = torch.meshgrid(z,y,x)
        # 输入变量为2.两个输出张量的行数为第一个输入张量的元素个数，列数为第二个输入张量的元素个数
        # 第一个输出张量填充第一个输入张量中的元素，各行元素相同
        # size:[1,1,64,64,64]
        xx = xx[None,None,:,:,:].cuda().float(); yy = yy[None,None,:,:,:].cuda().float(); zz = zz[None,None,:,:,:].cuda().float();

        #获得关节点的x y z坐标  joint_coord:[batch,42,3] x:[4,42,1,1,1]
        x = joint_coord[:,:,0,None,None,None]; y = joint_coord[:,:,1,None,None,None]; z = joint_coord[:,:,2,None,None,None];
        # 获得真实的值 热图
        heatmap = torch.exp(-(((xx-x)/cfg.sigma)**2)/2 -(((yy-y)/cfg.sigma)**2)/2 - (((zz-z)/cfg.sigma)**2)/2)
        heatmap = heatmap * 255
        # with torch.cuda.empty_cache():
        #     heatmap = torch.exp(-(((xx-x)/cfg.sigma)**2)/2 -(((yy-y)/cfg.sigma)**2)/2 - (((zz-z)/cfg.sigma)**2)/2)
        #     heatmap = heatmap * 255
        return heatmap
   
    def forward(self, inputs, targets, meta_info, mode):
        # [1, 3, 256, 256]
        input_img = inputs['img']
        #获得真实值

        # target_joint_coord, target_rel_root_depth, target_hand_type = targets['joint_coord'], targets['rel_root_depth'], targets['hand_type']
        # joint_valid, root_valid, hand_type_valid, inv_trans = meta_info['joint_valid'], meta_info['root_valid'], meta_info['hand_type_valid'], meta_info['inv_trans']
        
        batch_size = input_img.shape[0]

        # 通过ResNet网络得到图像特征
        img_feat= self.backbone_net(input_img)

        # img_feature = self.hourglass(input_img)
        # print(img_feature.shape)
        # merge_feature = torch.cat([img_feat,img_feature],1)

        merge_feature = img_feat
        # 得到左右手姿态位置 右手相对左手深度 手类型
        joint_heatmap_out, rel_root_depth_out, hand_type = self.pose_net(merge_feature)
        if mode == 'train':
            #获得真实的heatmap
            # target_joint_heatmap = self.render_gaussian_heatmap(target_joint_coord)
            target_joint_heatmap = self.render_gaussian_heatmap(targets['joint_coord'])

            #将loss值保存在字典中
            loss = {}
            # loss['joint_heatmap'] = self.joint_heatmap_loss(joint_heatmap_out, target_joint_heatmap, joint_valid)
            # loss['rel_root_depth'] = self.rel_root_depth_loss(rel_root_depth_out, target_rel_root_depth, root_valid)
            # loss['hand_type'] = self.hand_type_loss(hand_type, target_hand_type, hand_type_valid)
            loss['joint_heatmap'] = self.joint_heatmap_loss(joint_heatmap_out, target_joint_heatmap,meta_info['joint_valid'])
            loss['rel_root_depth'] = self.rel_root_depth_loss(rel_root_depth_out, targets['rel_root_depth'],meta_info['root_valid'])
            loss['hand_type'] = self.hand_type_loss(hand_type, targets['hand_type'], meta_info['hand_type_valid'])
            return loss
        elif mode == 'test':
            # 得到预测值并将其保存在out字典中
            out = {}
            # 选择第三个维度的最大值 分别对应值和索引 [batch,42,64,64]
            val_z, idx_z = torch.max(joint_heatmap_out,2)
            # [BATCH,42,64] val:值 idx:索引
            val_zy, idx_zy = torch.max(val_z,2)
            # [BATCH,42]
            val_zyx, joint_x = torch.max(val_zy,2)

            # [BATCH,42,1] 确定x轴的坐标
            joint_x = joint_x[:,:,None]
            # [BATCH,42,1]  gather(tensor,dim,index)-利用index来索引input特定位置的数值
            # 通过x轴找y轴坐标
            joint_y = torch.gather(idx_zy, 2, joint_x)
            # [BATCH,42,64]  ??
            joint_z = torch.gather(idx_z, 2, joint_y[:,:,:,None].repeat(1,1,1,cfg.output_hm_shape[1]))[:,:,0,:]
            # [BATCH,42,1]
            joint_z = torch.gather(joint_z, 2, joint_x)
            # [batch,42,3]
            joint_coord_out = torch.cat((joint_x, joint_y, joint_z),2).float()

            out['joint_coord'] = joint_coord_out
            out['rel_root_depth'] = rel_root_depth_out
            out['hand_type'] = hand_type
            # out['inv_trans'] = inv_trans
            # out['target_joint'] = target_joint_coord
            # out['joint_valid'] = joint_valid
            # out['hand_type_valid'] = hand_type_valid
            if 'inv_trans' in meta_info:
                out['inv_trans'] = meta_info['inv_trans']
            if 'joint_coord' in targets:
                out['target_joint'] = targets['joint_coord']
            if 'joint_valid' in meta_info:
                out['joint_valid'] = meta_info['joint_valid']
            if 'hand_type_valid' in meta_info:
                out['hand_type_valid'] = meta_info['hand_type_valid']
            return out

def init_weights(m):
    if type(m) == nn.ConvTranspose2d:
        # 正态
        nn.init.normal_(m.weight,std=0.001)
        # nn.init.kaiming_normal_(m.weight,mode='fan_out',nonlinearity="relu")
    elif type(m) == nn.Conv2d:
        nn.init.normal_(m.weight,std=0.001)
        # nn.init.kaiming_normal_(m.weight,mode='fan_out',nonlinearity="relu")
        # 初始化为常数
        nn.init.constant_(m.bias, 0)
    elif type(m) == nn.BatchNorm2d:
        nn.init.constant_(m.weight,1)
        nn.init.constant_(m.bias,0)
    elif type(m) == nn.Linear:
        nn.init.normal_(m.weight,std=0.01)
        # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity="relu")
        nn.init.constant_(m.bias,0)

def get_model(mode, joint_num):
    # 主干部分
    backbone_net = BackboneNet()
    # hourglass = StackedHourGlass(256,2,2,4,16)
    # 输出网络
    pose_net = PoseNet(joint_num)


    if mode == 'train':
        backbone_net.init_weights()
        # 递归的调用weights_init函数,遍历nn.Module的submodule作为参数
        # 常用来对模型的参数进行初始化
        # fn是对参数进行初始化的函数的句柄,fn以nn.Module或者自己定义的nn.Module的子类作为参数
        pose_net.apply(init_weights)
        # hourglass.apply(init_weights)
    # 文件顺序  module->model
    model = Model(backbone_net, pose_net)
    return model

if __name__ == "__main__":
    from thop import profile
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    backbone = BackboneNet().to(device)
    pose = PoseNet(21).to(device)
    pose.apply(init_weights)
    input = torch.randn(4,3,256,256).to(device)
    output = backbone(input)
    # parameters1 = sum(param.numel() for param in backbone.parameters())
    # parameters2 = sum(param.numel() for param in pose.parameters())
    # print((parameters1+parameters2)/10**6)
    output = backbone(input)
    flops1, params1 = profile(backbone, inputs=(input,))
    flops2, params2 = profile(pose, inputs=(output,))
    print("flops:", flops1 + flops2)
    print("params:", params1 + params2)
