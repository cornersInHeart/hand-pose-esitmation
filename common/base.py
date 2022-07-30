# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import os
import os.path as osp
import math
import time
import glob
import abc
from torch.utils.data import DataLoader
import torch.optim
import torchvision.transforms as transforms

from config import cfg
from dataset import Dataset
from timer import Timer
from logger import colorlogger
from torch.nn.parallel.data_parallel import DataParallel
from model import get_model

class Base(object):
    __metaclass__ = abc.ABCMeta

    def __init__(self, log_name='logs.txt'):
        
        self.cur_epoch = 0

        # timer 被子类继承
        self.tot_timer = Timer()  #total
        self.gpu_timer = Timer()  #gpu
        self.read_timer = Timer() #read

        # logger
        self.logger = colorlogger(cfg.log_dir, log_name=log_name)

    # 上述代码子类是约定俗称的实现这个方法，加上@abc.abstractmethod装饰器后严格控制子类必须实现这个方法
    @abc.abstractmethod
    def _make_batch_generator(self):
        return

    @abc.abstractmethod
    def _make_model(self):
        return


class Trainer(Base):
    
    def __init__(self):
        super(Trainer, self).__init__(log_name = 'train_logs.txt')
# 设置优化器optimizer
    def get_optimizer(self, model):
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
        return optimizer
# 设置学习率learing_rate
    def set_lr(self, epoch):
        # 不需要迭代减少lr,直接返回lr
        if len(cfg.lr_dec_epoch) == 0:
            return cfg.lr

        # 对于没到需要降低lr的epoch,直接跳出循环
        for e in cfg.lr_dec_epoch:
            if epoch < e:
                break
        # 当epoch=15/17,lr降低
        if epoch < cfg.lr_dec_epoch[-1]:
            idx = cfg.lr_dec_epoch.index(e)
            for g in self.optimizer.param_groups:
                g['lr'] = cfg.lr / (cfg.lr_dec_factor ** idx)
        else:
            for g in self.optimizer.param_groups:
                g['lr'] = cfg.lr / (cfg.lr_dec_factor ** len(cfg.lr_dec_epoch))
    # 将lr保存下来，以便获得当前批次的lr
    def get_lr(self):
        for g in self.optimizer.param_groups:
            cur_lr = g['lr']

        return cur_lr

    def _make_batch_generator(self, annot_subset):
        # data load and construct batch generator
        self.logger.info("Creating train dataset...")
        trainset_loader = Dataset(transforms.ToTensor(), "train", annot_subset)
        # dataloader一个可迭代对象，用iter()访问，不能用next()访问
        # iter(dataloader)返回的是一个迭代器，使用next访问| 也可使用for inputs,labels in dataloaders访问
        # 每次处理一个batch的数据
        batch_generator = DataLoader(dataset=trainset_loader, batch_size=cfg.num_gpus*cfg.train_batch_size, shuffle=True, num_workers=cfg.num_thread, pin_memory=True)
        
        self.joint_num = trainset_loader.joint_num
        #向上取整 每块gpu每个批次迭代的次数
        self.itr_per_epoch = math.ceil(trainset_loader.__len__() / cfg.num_gpus / cfg.train_batch_size)
        self.batch_generator = batch_generator

    def _make_model(self):
        # prepare network
        self.logger.info("Creating graph and optimizer...")

        model = get_model('train', self.joint_num)
        # 将模型放入gpu中 调用多个GPU
        model = DataParallel(model).cuda()
        optimizer = self.get_optimizer(model)
        if cfg.continue_train:
            start_epoch, model, optimizer = self.load_model(model, optimizer)
        else:
            start_epoch = 0
        # model.train()是保证BN层用每一批数据的均值和方差，而model.eval()是保证BN用全部训练数据的均值和方差
        # model.train()是随机取一部分网络连接来训练更新参数，而model.eval()是利用到了所有网络连接
        model.train()
        
        self.start_epoch = start_epoch
        # 将model赋给self.model,使得可以直接调用get_model
        self.model = model
        self.optimizer = optimizer
         
    def save_model(self, state, epoch):
        file_path = osp.join(cfg.model_dir,'snapshot_{}.pth.tar'.format(str(epoch)))
        #保存模型
        torch.save(state, file_path)
        self.logger.info("Write snapshot into {}".format(file_path))

    def load_model(self, model, optimizer):
        model_file_list = glob.glob(osp.join(cfg.model_dir,'*.pth.tar'))
        # 获取当前运行到的批次
        cur_epoch = max([int(file_name[file_name.find('snapshot_') + 9 : file_name.find('.pth.tar')]) for file_name in model_file_list])
        # 获取已经保存的模型
        model_path = osp.join(cfg.model_dir, 'snapshot_' + str(cur_epoch) + '.pth.tar')
        self.logger.info('Load checkpoint from {}'.format(model_path))
        # 加载之前的模型
        ckpt = torch.load(model_path)
        # 当前运行批次+1
        start_epoch = ckpt['epoch'] + 1
        # 加载模型的参数
        # model.load_state_dict(ckpt['network'], strict=False)
        model.load_state_dict(ckpt['network'])
        try:
            # 加载优化器的参数
            optimizer.load_state_dict(ckpt['optimizer'])
        except:
            pass

        return start_epoch, model, optimizer


class Tester(Base):
    
    def __init__(self, test_epoch):
        self.test_epoch = int(test_epoch)
        super(Tester, self).__init__(log_name = 'test_logs.txt')

    def _make_batch_generator(self, test_set, annot_subset):
        # data load and construct batch generator
        self.logger.info("Creating " + test_set + " dataset...")
        testset_loader = Dataset(transforms.ToTensor(), test_set, annot_subset)
        batch_generator = DataLoader(dataset=testset_loader, batch_size=cfg.num_gpus*cfg.test_batch_size, shuffle=False, num_workers=cfg.num_thread, pin_memory=True)
        
        self.joint_num = testset_loader.joint_num
        self.batch_generator = batch_generator
        print("ok")
        self.testset = testset_loader
    
    def _make_model(self):
        # 取出保存该批次的模型路径
        model_path = os.path.join(cfg.model_dir, 'snapshot_%d.pth.tar' % self.test_epoch)
        assert os.path.exists(model_path), 'Cannot find model at ' + model_path
        self.logger.info('Load checkpoint from {}'.format(model_path))
        
        # prepare network
        self.logger.info("Creating graph...")
        model = get_model('test', self.joint_num)
        model = DataParallel(model).cuda()
        # 读取之前保存的网络模型参数
        ckpt = torch.load(model_path)
        # model.load_state_dict(ckpt['network'], strict=False)
        model.load_state_dict(ckpt['network'])
        # 切换到test模式，固定BN和dropout层，使得偏置参数不发生变化。因为当batchsize小时，如果没有固定，会对图像的失真有很大的影响。
        model.eval()

        self.model = model

    def _evaluate(self, preds):
        # 指向testset_loader
        self.testset.evaluate(preds)

