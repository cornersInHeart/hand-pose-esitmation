# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import os
import cv2
import numpy as np
from config import cfg
import random
import math
import matplotlib.pyplot as plt

def load_img(path, order='RGB'):
    
    # load cv2.  IMREAD_COLOR：默认参数，读入一副彩色图片，忽略alpha通道
    # cv2.IMREAD_GRAYSCALE：读入灰度图片
    # cv2.IMREAD_UNCHANGED：顾名思义，读入完整图片，包括alpha通道
    img = cv2.imread(path, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
    if not isinstance(img, np.ndarray):
        raise IOError("Fail to read %s" % path)

    if order == 'RGB':
        # 实现RGB与BGR的转换  cv2把图片读取后是把图片读成BGR形式的，plt则是读成RGB形式
        img = img[:,:,::-1].copy()
    
    img = img.astype(np.float32)
    return img

def load_skeleton(path, joint_num): # seleton:骨架
    # load joint info (name, parent_id)
    skeleton = [{} for _ in range(joint_num)]  # [{name parent_id child_id}..]
    with open(path) as fp:
        for line in fp:
            if line[0] == '#': continue
            splitted = line.split(' ')
            joint_name, joint_id, joint_parent_id = splitted
            joint_id, joint_parent_id = int(joint_id), int(joint_parent_id)
            # joint_id作为索引去查询第i个关节点的相关信息
            skeleton[joint_id]['name'] = joint_name
            skeleton[joint_id]['parent_id'] = joint_parent_id
    # save child_id
    for i in range(len(skeleton)):
        joint_child_id = []
        for j in range(len(skeleton)):
            if skeleton[j]['parent_id'] == i:
                joint_child_id.append(j)
        # 每次遍历后将列表joint_child_id存储到相应skeleton
        skeleton[i]['child_id'] = joint_child_id
    
    return skeleton

# 进行数据增强的配置
def get_aug_config():
    trans_factor = 0.15
    scale_factor = 0.25
    rot_factor = 45
    color_factor = 0.2
    
    # 平移 x,y
    trans = [np.random.uniform(-trans_factor, trans_factor), np.random.uniform(-trans_factor, trans_factor)]
    # 缩放
    scale = np.clip(np.random.randn(), -1.0, 1.0) * scale_factor + 1.0
    # 旋转
    rot = np.clip(np.random.randn(), -2.0,2.0) * rot_factor if random.random() <= 0.6 else 0
    # 水平反转
    do_flip = random.random() <= 0.5
    c_up = 1.0 + color_factor
    c_low = 1.0 - color_factor
    # 颜色抖动 三个通道
    color_scale = np.array([random.uniform(c_low, c_up), random.uniform(c_low, c_up), random.uniform(c_low, c_up)])

    return trans, scale, rot, do_flip, color_scale

def augmentation(img, bbox, joint_coord, joint_valid, hand_type, mode, joint_type):
    img = img.copy(); 
    joint_coord = joint_coord.copy(); 
    hand_type = hand_type.copy();

    original_img_shape = img.shape
    joint_num = len(joint_coord)

    # train  use data augmentation
    if mode == 'train':
        trans, scale, rot, do_flip, color_scale = get_aug_config()
    # test|val:use normal data
    else:
        trans, scale, rot, do_flip, color_scale = [0,0], 1.0, 0.0, False, np.array([1,1,1])

    # bbox:(x,y,w,h) 将bounding box进行平移
    bbox[0] = bbox[0] + bbox[2] * trans[0]
    bbox[1] = bbox[1] + bbox[3] * trans[1]
    img, trans, inv_trans = generate_patch_image(img, bbox, do_flip, scale, rot, cfg.input_img_shape)
    # 截取 0-255
    img = np.clip(img * color_scale[None,None,:], 0, 255)
    
    if do_flip:
        # 进行图片翻转
        joint_coord[:,0] = original_img_shape[1] - joint_coord[:,0] - 1
        joint_coord[joint_type['right']], joint_coord[joint_type['left']] = joint_coord[joint_type['left']].copy(), joint_coord[joint_type['right']].copy()
        joint_valid[joint_type['right']], joint_valid[joint_type['left']] = joint_valid[joint_type['left']].copy(), joint_valid[joint_type['right']].copy()
        hand_type[0], hand_type[1] = hand_type[1].copy(), hand_type[0].copy()
    for i in range(joint_num):
        joint_coord[i,:2] = trans_point2d(joint_coord[i,:2], trans)
        joint_valid[i] = joint_valid[i] * (joint_coord[i,0] >= 0) * (joint_coord[i,0] < cfg.input_img_shape[1]) * (joint_coord[i,1] >= 0) * (joint_coord[i,1] < cfg.input_img_shape[0])

    return img, joint_coord, joint_valid, hand_type, inv_trans

def transform_input_to_output_space(joint_coord, joint_valid, rel_root_depth, root_valid, root_joint_idx, joint_type):
    # transform to output heatmap space
    joint_coord = joint_coord.copy(); joint_valid = joint_valid.copy()

    # 变换到输出空间
    joint_coord[:,0] = joint_coord[:,0] / cfg.input_img_shape[1] * cfg.output_hm_shape[2]
    joint_coord[:,1] = joint_coord[:,1] / cfg.input_img_shape[0] * cfg.output_hm_shape[1]

    joint_coord[joint_type['right'],2] = joint_coord[joint_type['right'],2] - joint_coord[root_joint_idx['right'],2]
    joint_coord[joint_type['left'],2]  = joint_coord[joint_type['left'],2] - joint_coord[root_joint_idx['left'],2]
  
    joint_coord[:,2] = (joint_coord[:,2] / (cfg.bbox_3d_size/2) + 1)/2. * cfg.output_hm_shape[0]
    joint_valid = joint_valid * ((joint_coord[:,2] >= 0) * (joint_coord[:,2] < cfg.output_hm_shape[0])).astype(np.float32)
    rel_root_depth = (rel_root_depth / (cfg.bbox_3d_size_root/2) + 1)/2. * cfg.output_root_hm_shape
    root_valid = root_valid * ((rel_root_depth >= 0) * (rel_root_depth < cfg.output_root_hm_shape)).astype(np.float32)
    
    return joint_coord, joint_valid, rel_root_depth, root_valid

def get_bbox(joint_img, joint_valid):
    # 取出所有对关节进行注释的x,y
    x_img = joint_img[:,0][joint_valid==1]; y_img = joint_img[:,1][joint_valid==1];
    xmin = min(x_img); ymin = min(y_img); xmax = max(x_img); ymax = max(y_img);

    x_center = (xmin+xmax)/2.; width = xmax-xmin;
    xmin = x_center - 0.5*width*1.2
    xmax = x_center + 0.5*width*1.2
    
    y_center = (ymin+ymax)/2.; height = ymax-ymin;
    ymin = y_center - 0.5*height*1.2
    ymax = y_center + 0.5*height*1.2

    # 拼接四个值得到bbox
    bbox = np.array([xmin, ymin, xmax-xmin, ymax-ymin]).astype(np.float32)
    return bbox

# 将边界框缩放到相同比例
def  process_bbox(bbox, original_img_shape):

    # aspect ratio preserving bbox [xmin, ymin, width, height]
    w = bbox[2]
    h = bbox[3]
    c_x = bbox[0] + w/2.
    c_y = bbox[1] + h/2.
    # input_img_shape = (256, 256)
    # 设置纵横比
    aspect_ratio = cfg.input_img_shape[1]/cfg.input_img_shape[0]
    # 缩放到相同比例
    if w > aspect_ratio * h:
        h = w / aspect_ratio
    elif w < aspect_ratio * h:
        w = h * aspect_ratio
    # 得到新的bbox的值
    bbox[2] = w*1.25
    bbox[3] = h*1.25
    bbox[0] = c_x - bbox[2]/2.
    bbox[1] = c_y - bbox[3]/2.

    return bbox

def generate_patch_image(cvimg, bbox, do_flip, scale, rot, out_shape):
    # (512, 334, 3)
    img = cvimg.copy()
    img_height, img_width, img_channels = img.shape
    # 中心点 宽高
    bb_c_x = float(bbox[0] + 0.5*bbox[2])
    bb_c_y = float(bbox[1] + 0.5*bbox[3])
    bb_width = float(bbox[2])
    bb_height = float(bbox[3])

    if do_flip:
        #翻转
        img = img[:, ::-1, :]
        bb_c_x = img_width - bb_c_x - 1  # ？？

    # (2,3)
    trans = gen_trans_from_patch_cv(bb_c_x, bb_c_y, bb_width, bb_height, out_shape[1], out_shape[0], scale, rot)
    # cv2.warpAffine(src, M, dsize[, dst[, flags[, borderMode[, borderValue]]]]) → dst  得到的trans对原始图像进行变换
    # 此方法经过放射变化后，可能会出现"黑边" borderValue=(255,255,255)设置为白色
    # 仿射变函数
    img_patch = cv2.warpAffine(img, trans, (int(out_shape[1]), int(out_shape[0])), flags=cv2.INTER_LINEAR)
    # (256,256,3)
    # plt.imshow(img_patch)
    img_patch = img_patch.astype(np.float32)
    # plt.show()
    # (2,3)
    inv_trans = gen_trans_from_patch_cv(bb_c_x, bb_c_y, bb_width, bb_height, out_shape[1], out_shape[0], scale, rot, inv=True)
    return img_patch, trans, inv_trans

#平面坐标变换 pt_2d:array
def rotate_2d(pt_2d, rot_rad):
    x = pt_2d[0]
    y = pt_2d[1]
    sn, cs = np.sin(rot_rad), np.cos(rot_rad)
    xx = x * cs - y * sn
    yy = x * sn + y * cs
    return np.array([xx, yy], dtype=np.float32)


def gen_trans_from_patch_cv(c_x, c_y, src_width, src_height, dst_width, dst_height, scale, rot, inv=False):
    # augment size with scale 扩大规模
    src_w = src_width * scale
    src_h = src_height * scale
    src_center = np.array([c_x, c_y], dtype=np.float32)

    # augment rotation  角度
    rot_rad = np.pi * rot / 180
    # 得到变换前的平面坐标
    src_downdir = rotate_2d(np.array([0, src_h * 0.5], dtype=np.float32), rot_rad)
    src_rightdir = rotate_2d(np.array([src_w * 0.5, 0], dtype=np.float32), rot_rad)
    # ？？
    dst_w = dst_width
    dst_h = dst_height
    dst_center = np.array([dst_w * 0.5, dst_h * 0.5], dtype=np.float32)
    dst_downdir = np.array([0, dst_h * 0.5], dtype=np.float32)
    dst_rightdir = np.array([dst_w * 0.5, 0], dtype=np.float32)

    src = np.zeros((3, 2), dtype=np.float32)
    src[0, :] = src_center
    src[1, :] = src_center + src_downdir
    src[2, :] = src_center + src_rightdir

    dst = np.zeros((3, 2), dtype=np.float32)
    dst[0, :] = dst_center
    dst[1, :] = dst_center + dst_downdir
    dst[2, :] = dst_center + dst_rightdir
    # src,dst 变换前后两个对应的点
    if inv:
        # 进行仿射变换 从输入的点到输出的点 trans：根据三个对应点求出的仿射变换矩阵
        trans = cv2.getAffineTransform(np.float32(dst), np.float32(src))
    else:
        trans = cv2.getAffineTransform(np.float32(src), np.float32(dst))

    trans = trans.astype(np.float32)
    return trans

def trans_point2d(pt_2d, trans):
    src_pt = np.array([pt_2d[0], pt_2d[1], 1.]).T
    dst_pt = np.dot(trans, src_pt)
    return dst_pt[0:2]


