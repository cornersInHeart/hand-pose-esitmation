# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import numpy as np
import torch
import torch.utils.data
import cv2
from glob import glob
import os.path as osp
from config import cfg
from utils.preprocessing import load_img, load_skeleton, get_bbox, process_bbox, augmentation, transform_input_to_output_space, trans_point2d
from utils.transforms import world2cam, cam2pixel, pixel2cam
from utils.vis import vis_keypoints, vis_3d_keypoints
from PIL import Image, ImageDraw
import random
import json
import math
from pycocotools.coco import COCO
import scipy.io as sio

class Dataset(torch.utils.data.Dataset):
    def __init__(self, transform, mode, annot_subset):
        self.mode = mode # train, test, val
        # 手势标注类型
        self.annot_subset = annot_subset # all, human_annot, machine_annot

        ##  ---
        self.img_path = '../data/InterHand2.6M/images'
        self.annot_path = '../data/InterHand2.6M/annotations'
        if self.annot_subset == 'machine_annot' and self.mode == 'val':
            self.rootnet_output_path = '../data/InterHand2.6M/rootnet_output/rootnet_interhand2.6m_output_machine_annot_val.json'
        else:
            self.rootnet_output_path = '../data/InterHand2.6M/rootnet_output/rootnet_interhand2.6m_output_all_test.json'
        self.transform = transform # 对图像进行变化处理
        self.joint_num = 21 # single hand 每个手标注21个关节点
        self.root_joint_idx = {'right': 20, 'left': 41}
        self.joint_type = {'right': np.arange(0,self.joint_num), 'left': np.arange(self.joint_num,self.joint_num*2)} # 标号
        # 加载关节的信息 name parent_id child_id
        self.skeleton = load_skeleton(osp.join(self.annot_path, 'skeleton.txt'), self.joint_num*2)
        
        self.datalist = []
        # SH表示单手 IH表示交互手
        self.datalist_sh = []
        self.datalist_ih = []
        self.sequence_names = []
        
        # load annotation
        print("Load annotation from  " + osp.join(self.annot_path, self.annot_subset))
        # loads COCO annotation file and prepare data structures
        # db = COCO(osp.join(self.annot_path, self.annot_subset, 'InterHand2.6M_' + self.mode + '_data.json'))
        # -----
        db = COCO(osp.join(self.annot_path, self.annot_subset, 'InterHand2.6M_' + self.mode + '_data.json'))
        # 加载json文件
        with open(osp.join(self.annot_path, self.annot_subset, 'InterHand2.6M_' + self.mode + '_camera.json')) as f:
            cameras = json.load(f)
        with open(osp.join(self.annot_path, self.annot_subset, 'InterHand2.6M_' + self.mode + '_joint_3d.json')) as f:
            joints = json.load(f)

        if (self.mode == 'val' or self.mode == 'test') and cfg.trans_test == 'rootnet':
            print("Get bbox and root depth from " + self.rootnet_output_path)
            rootnet_result = {}
            with open(self.rootnet_output_path) as f:
                annot = json.load(f)
            for i in range(len(annot)):
                rootnet_result[str(annot[i]['annot_id'])] = annot[i]
        else:
            print("Get bbox and root depth from groundtruth annotation")

        # {ID:ann}
        for aid in db.anns.keys():
            ann = db.anns[aid]
            image_id = ann['image_id']
            # Load imgs with the specified ids 从image_id获得以下image的字典信息
            img = db.loadImgs(image_id)[0]
 
            capture_id = img['capture']
            seq_name = img['seq_name']
            cam = img['camera']
            frame_idx = img['frame_idx']
            ##  --- ../data/InterHand2.6M/images+mode+'file_name'
            # img_path = osp.join(self.img_path, self.mode, img['file_name'])
            img_path = osp.join(self.img_path, self.mode, img['file_name'])

            # 相机矩阵 相机旋转矩阵
            campos, camrot = np.array(cameras[str(capture_id)]['campos'][str(cam)], dtype=np.float32), np.array(cameras[str(capture_id)]['camrot'][str(cam)], dtype=np.float32)
            # 焦距 像主点
            focal, princpt = np.array(cameras[str(capture_id)]['focal'][str(cam)], dtype=np.float32), np.array(cameras[str(capture_id)]['princpt'][str(cam)], dtype=np.float32)
            # 世界坐标
            joint_world = np.array(joints[str(capture_id)][str(frame_idx)]['world_coord'], dtype=np.float32)
            # 进行坐标转换
            joint_cam = world2cam(joint_world.transpose(1,0), camrot, campos.reshape(3,1)).transpose(1,0)
            joint_img = cam2pixel(joint_cam, focal, princpt)[:,:2]

            joint_valid = np.array(ann['joint_valid'],dtype=np.float32).reshape(self.joint_num*2)
            # if root is not valid -> root-relative 3D pose is also not valid. Therefore, mark all joints as invalid
            joint_valid[self.joint_type['right']] *= joint_valid[self.root_joint_idx['right']]
            joint_valid[self.joint_type['left']] *= joint_valid[self.root_joint_idx['left']]
            hand_type = ann['hand_type'] # str('right', 'left', 'interacting')
            hand_type_valid = np.array((ann['hand_type_valid']), dtype=np.float32)
            
            if (self.mode == 'val' or self.mode == 'test') and cfg.trans_test == 'rootnet':
                bbox = np.array(rootnet_result[str(aid)]['bbox'],dtype=np.float32)
                # 获取绝对深度
                abs_depth = {'right': rootnet_result[str(aid)]['abs_depth'][0], 'left': rootnet_result[str(aid)]['abs_depth'][1]}
            else:
                img_width, img_height = img['width'], img['height']
                bbox = np.array(ann['bbox'],dtype=np.float32) # xmin, ymin, width, height
                bbox = process_bbox(bbox, (img_height, img_width))
                abs_depth = {'right': joint_cam[self.root_joint_idx['right'],2], 'left': joint_cam[self.root_joint_idx['left'],2]}

            #相机参数
            cam_param = {'focal': focal, 'princpt': princpt}
            #关节参数
            joint = {'cam_coord': joint_cam, 'img_coord': joint_img, 'valid': joint_valid}
            # 将相应的信息存入data中
            data = {'img_path': img_path, 'seq_name': seq_name, 'cam_param': cam_param, 'bbox': bbox, 'joint': joint, 'hand_type': hand_type, 'hand_type_valid': hand_type_valid, 'abs_depth': abs_depth, 'file_name': img['file_name'], 'capture': capture_id, 'cam': cam, 'frame': frame_idx}
            if hand_type == 'right' or hand_type == 'left':
                # # ----
                # if osp.exists(data['img_path']):
                self.datalist_sh.append(data)
            else:
                ## ----
                # if osp.exists(data['img_path']):
                self.datalist_ih.append(data)
            if seq_name not in self.sequence_names:
                self.sequence_names.append(seq_name)

        # 合并list
        self.datalist = self.datalist_sh + self.datalist_ih

        # #
        # np.random.shuffle(self.datalist)
        # if self.mode=="train":
        #     self.datalist=self.datalist[:]
        # elif self.mode=="test":
        #     self.datalist = self.datalist[:]

        print('Number of annotations in single hand sequences: ' + str(len(self.datalist_sh)))
        print('Number of annotations in interacting hand sequences: ' + str(len(self.datalist_ih)))

    # 手的类型数据化
    def handtype_str2array(self, hand_type):
        if hand_type == 'right':
            return np.array([1,0], dtype=np.float32)
        elif hand_type == 'left':
            return np.array([0,1], dtype=np.float32)
        elif hand_type == 'interacting':
            return np.array([1,1], dtype=np.float32)
        else:
            assert 0, print('Not supported hand type: ' + hand_type)

    #获取数据集的长度
    def __len__(self):
        return len(self.datalist)

    #获取到某条数据
    def __getitem__(self, idx):
        # 取某一个值进行计算
        data = self.datalist[idx]
        ## ----
        img_path, bbox, joint, hand_type, hand_type_valid = data['img_path'], data['bbox'], data['joint'], data['hand_type'], data['hand_type_valid']
        joint_cam = joint['cam_coord'].copy(); joint_img = joint['img_coord'].copy(); joint_valid = joint['valid'].copy();
        hand_type = self.handtype_str2array(hand_type) # array
        # [42,3]
        joint_coord = np.concatenate((joint_img, joint_cam[:,2,None]),1)
        
        # image load
        img = load_img(img_path)
        # augmentation
        img, joint_coord, joint_valid, hand_type, inv_trans = augmentation(img, bbox, joint_coord, joint_valid, hand_type, self.mode, self.joint_type)
        # 计算左右手的相对深度 reshape(1)将其变成一维数组
        rel_root_depth = np.array([joint_coord[self.root_joint_idx['left'],2] - joint_coord[self.root_joint_idx['right'],2]],dtype=np.float32).reshape(1)
        root_valid = np.array([joint_valid[self.root_joint_idx['right']] * joint_valid[self.root_joint_idx['left']]],dtype=np.float32).reshape(1) if hand_type[0]*hand_type[1] == 1 else np.zeros((1),dtype=np.float32)
        # transform to output heatmap space
        joint_coord, joint_valid, rel_root_depth, root_valid = transform_input_to_output_space(joint_coord, joint_valid, rel_root_depth, root_valid, self.root_joint_idx, self.joint_type)
        img = self.transform(img.astype(np.float32))/255.
        
        inputs = {'img': img}
        #真实数据值 关节坐标 根关节相对深度 手的类型
        targets = {'joint_coord': joint_coord, 'rel_root_depth': rel_root_depth, 'hand_type': hand_type}
        meta_info = {'joint_valid': joint_valid, 'root_valid': root_valid, 'hand_type_valid': hand_type_valid, 'inv_trans': inv_trans, 'capture': int(data['capture']), 'cam': int(data['cam']), 'frame': int(data['frame'])}
        return inputs, targets, meta_info

    def evaluate(self, preds):

        print() 
        print('Evaluation start...')

        gts = self.datalist
        preds_joint_coord, preds_rel_root_depth, preds_hand_type, inv_trans = preds['joint_coord'], preds['rel_root_depth'], preds['hand_type'], preds['inv_trans']
        # 当真实数据集的长度与预测数据集长度不等时报错
        assert len(gts) == len(preds_joint_coord)
        sample_num = len(gts)

        # 获取单手和双手的mpjpe评估-关节的平均误差
        mpjpe_sh = [[] for _ in range(self.joint_num*2)]
        mpjpe_ih = [[] for _ in range(self.joint_num*2)]
        # 相对深度的平均误差
        mrrpe = []
        # 预测正确手的数量  总的手的数量
        acc_hand_cls = 0; hand_cls_cnt = 0;
        for n in range(sample_num):
            # real
            data = gts[n]
            bbox, cam_param, joint, gt_hand_type, hand_type_valid = data['bbox'], data['cam_param'], data['joint'], data['hand_type'], data['hand_type_valid']
            focal = cam_param['focal']
            princpt = cam_param['princpt']
            gt_joint_coord = joint['cam_coord']
            joint_valid = joint['valid']


            # restore pred xy coordinates to original image space
            pred_joint_coord_img = preds_joint_coord[n].copy()
            pred_joint_coord_img[:,0] = pred_joint_coord_img[:,0]/cfg.output_hm_shape[2]*cfg.input_img_shape[1]
            pred_joint_coord_img[:,1] = pred_joint_coord_img[:,1]/cfg.output_hm_shape[1]*cfg.input_img_shape[0]
            for j in range(self.joint_num*2):
                # final 3D hand pose
                pred_joint_coord_img[j,:2] = trans_point2d(pred_joint_coord_img[j,:2],inv_trans[n])
            # restore pred depth to original camera space
            pred_joint_coord_img[:,2] = (pred_joint_coord_img[:,2]/cfg.output_hm_shape[0] * 2 - 1) * (cfg.bbox_3d_size/2)  # ?
 
            # mrrpe 平均相对根位置误差
            if gt_hand_type == 'interacting' and joint_valid[self.root_joint_idx['left']] and joint_valid[self.root_joint_idx['right']]:
                pred_rel_root_depth = (preds_rel_root_depth[n]/cfg.output_root_hm_shape * 2 - 1) * (cfg.bbox_3d_size_root/2)# ?

                pred_left_root_img = pred_joint_coord_img[self.root_joint_idx['left']].copy()

                pred_left_root_img[2] += data['abs_depth']['right'] + pred_rel_root_depth
                pred_left_root_cam = pixel2cam(pred_left_root_img[None,:], focal, princpt)[0]

                pred_right_root_img = pred_joint_coord_img[self.root_joint_idx['right']].copy()
                pred_right_root_img[2] += data['abs_depth']['right']
                pred_right_root_cam = pixel2cam(pred_right_root_img[None,:], focal, princpt)[0]

                # 真实与预测的差值
                pred_rel_root = pred_left_root_cam - pred_right_root_cam
                gt_rel_root = gt_joint_coord[self.root_joint_idx['left']] - gt_joint_coord[self.root_joint_idx['right']]
                mrrpe.append(float(np.sqrt(np.sum((pred_rel_root - gt_rel_root)**2))))

           
            # add root joint depth
            pred_joint_coord_img[self.joint_type['right'],2] += data['abs_depth']['right']
            pred_joint_coord_img[self.joint_type['left'],2] += data['abs_depth']['left']

            # back project to camera coordinate system
            pred_joint_coord_cam = pixel2cam(pred_joint_coord_img, focal, princpt)

            # root joint alignment
            for h in ('right', 'left'):
                pred_joint_coord_cam[self.joint_type[h]] = pred_joint_coord_cam[self.joint_type[h]] - pred_joint_coord_cam[self.root_joint_idx[h],None,:]
                gt_joint_coord[self.joint_type[h]] = gt_joint_coord[self.joint_type[h]] - gt_joint_coord[self.root_joint_idx[h],None,:]
            
            # mpjpe:mean per joint position error 欧氏距离
            for j in range(self.joint_num*2):
                if joint_valid[j]:
                    if gt_hand_type == 'right' or gt_hand_type == 'left':
                        mpjpe_sh[j].append(np.sqrt(np.sum((pred_joint_coord_cam[j] - gt_joint_coord[j])**2)))
                    else: # interacting hand
                        mpjpe_ih[j].append(np.sqrt(np.sum((pred_joint_coord_cam[j] - gt_joint_coord[j])**2)))

            # handedness accuray
            if hand_type_valid:
                if gt_hand_type == 'right' and preds_hand_type[n][0] > 0.5 and preds_hand_type[n][1] < 0.5:
                    acc_hand_cls += 1
                elif gt_hand_type == 'left' and preds_hand_type[n][0] < 0.5 and preds_hand_type[n][1] > 0.5:
                    acc_hand_cls += 1
                elif gt_hand_type == 'interacting' and preds_hand_type[n][0] > 0.5 and preds_hand_type[n][1] > 0.5:
                    acc_hand_cls += 1
                hand_cls_cnt += 1

            vis = False
            if vis:
                img_path = data['img_path']
                img_str = "_".join(img_path.split("/")[-4:])
                cvimg = cv2.imread(img_path, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
                _img = cvimg[:,:,::-1].transpose(2,0,1)
                # 关节点坐标
                vis_kps = pred_joint_coord_img.copy()
                # 关节点是否用来注释去进行估计
                vis_valid = joint_valid.copy()
                capture = str(data['capture'])
                cam = str(data['cam'])
                frame = str(data['frame'])
                # 输出结果路径
                filename = 'out_' + str(n) + '_' + gt_hand_type + '||' +img_str
                vis_keypoints(_img, vis_kps, vis_valid, self.skeleton, filename)

            vis = False
            if vis:
                # 输出3d坐标图像路径
                filename = 'out_' + str(n) + '_3d||'+ img_str
                vis_3d_keypoints(pred_joint_coord_cam, joint_valid, self.skeleton, filename)

        evalute_file = "../output/log/evalute.txt"
        fp = open(evalute_file,"a")
        # 输出handedness的准确率
        if hand_cls_cnt > 0: print('Handedness accuracy: ' + str(acc_hand_cls / hand_cls_cnt))
        print(len(mrrpe),sum(mrrpe))
        fp.write('Handedness accuracy: ' + str(acc_hand_cls / hand_cls_cnt)+"\n")
        if len(mrrpe) > 0: print('MRRPE: ' + str(sum(mrrpe)/len(mrrpe)))
        print()
        fp.write('MRRPE: ' + str(sum(mrrpe)/len(mrrpe))+"\n")
 
        tot_err = []
        eval_summary = 'MPJPE for each joint: \n'
        # 评估单手和交互手MPJPE
        for j in range(self.joint_num*2):
            # mpjpe_sh:list np.stack()可以将其在axis=0增加一维
            # np.concatenate 数组拼接，默认第零维
            # print(len(mpjpe_ih[j]),len(mpjpe_sh[j]))
            tot_err_j = np.mean(np.concatenate((np.stack(mpjpe_sh[j]), np.stack(mpjpe_ih[j]))))
            joint_name = self.skeleton[j]['name']
            eval_summary += (joint_name + ': %.2f, ' % tot_err_j)
            tot_err.append(tot_err_j)
        print(eval_summary)
        print('MPJPE for all hand sequences: %.2f' % (np.mean(tot_err)))
        fp.write(eval_summary+"\n")
        fp.write('MPJPE for all hand sequences: %.2f' % (np.mean(tot_err))+"\n")
        print()

        eval_summary = 'MPJPE for each joint: \n'
        #评估单手MPJPE
        for j in range(self.joint_num*2):
            mpjpe_sh[j] = np.mean(np.stack(mpjpe_sh[j]))
            joint_name = self.skeleton[j]['name']
            eval_summary += (joint_name + ': %.2f, ' % mpjpe_sh[j])
        print(eval_summary)
        print('MPJPE for single hand sequences: %.2f' % (np.mean(mpjpe_sh)))
        fp.write(eval_summary + "\n")
        fp.write('MPJPE for single hand sequences: %.2f' % (np.mean(mpjpe_sh))+"\n")
        print()

        eval_summary = 'MPJPE for each joint: \n'
        #评估交互手的MPJPE
        for j in range(self.joint_num*2):
            mpjpe_ih[j] = np.mean(np.stack(mpjpe_ih[j]))
            joint_name = self.skeleton[j]['name']
            eval_summary += (joint_name + ': %.2f, ' % mpjpe_ih[j])
        print(eval_summary)
        print('MPJPE for interacting hand sequences: %.2f' % (np.mean(mpjpe_ih)))
        fp.write(eval_summary + "\n")
        fp.write('MPJPE for interacting hand sequences: %.2f' % (np.mean(mpjpe_ih))+"\n")
        fp.close()


