# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#blog.csdn.net/qq_32642107/article/details/100565434:DataLoader worker (pid ****) is killed by signal: Killed

import argparse
from config import cfg
import torch
from base import Trainer
import torch.backends.cudnn as cudnn
import json
from tqdm import tqdm

# python train.py --gpu 0 --annot_subset machine_annot


#argparse是一个Python模块：命令行选项、参数和子命令解析器。 编写用户友好的命令行接口
def parse_args():
    # 创建解析器  使用argparse的第一步是创建一个ArgumentParser对象。
    parser = argparse.ArgumentParser()
    # 添加参数  添加程序参数信息调用add_argument()完成的
    # 在 add_argument 前，给属性名之前加上“- -”，就能将之变为可选参数。
    parser.add_argument('--gpu', type=str, dest='gpu_ids')
    # type-命令行参数应当被转换成的类型 action:当参数在命令行中出现时使用的动作基本类型 dest:添加到parse_args()所返回对象的属性名
    # parser.add_argument('--continue',type=str,dest='continue_train', action='store_true')
    parser.add_argument('--continue',type=str,dest='continue_train')
    # 这些信息在 parse_args() 调用时被存储和使用
    parser.add_argument('--annot_subset', type=str, dest='annot_subset')
    # 解析函数
    args = parser.parse_args()

    if not args.gpu_ids:
        assert 0, "Please set propoer gpu ids"

    if '-' in args.gpu_ids:
        gpus = args.gpu_ids.split('-')
        gpus[0] = int(gpus[0])
        gpus[1] = int(gpus[1]) + 1
        args.gpu_ids = ','.join(map(lambda x: str(x), list(range(*gpus))))

    return args

def main():
    
    # argument parse and create log
    args = parse_args()
    #设置gpu运行的相关信息
    cfg.set_args(args.gpu_ids, args.continue_train)
    cudnn.benchmark = True

    if cfg.dataset == 'InterHand2.6M':
        assert args.annot_subset, "Please set proper annotation subset. Select one of all, human_annot, machine_annot"
    else:
        args.annot_subset = None
    # 生成一个Trainer的对象
    trainer = Trainer()
    # 批量生成数据
    trainer._make_batch_generator(args.annot_subset)
    # 准备model optimizer loss
    trainer._make_model()
    
    # train
    for epoch in range(trainer.start_epoch, cfg.end_epoch):
        # set learning_rate
        trainer.set_lr(epoch)
        #total:start time
        trainer.tot_timer.tic()
        #read:start time
        trainer.read_timer.tic()

        total_loss={}
        for itr, (inputs, targets, meta_info) in enumerate(tqdm(trainer.batch_generator)):
            trainer.read_timer.toc()
            # gpu run in each iter:start time
            trainer.gpu_timer.tic()

            # forward
            trainer.optimizer.zero_grad()
            # 调用get_model函数得到需要的model和loss,并计算损失
            loss = trainer.model(inputs, targets, meta_info, 'train')
            loss = {k:loss[k].mean() for k in loss}

            for key,value in loss.items():
                if key not in total_loss:
                    total_loss[key] = 0
                total_loss[key] += value.detach()
            # backward
            sum(loss[k] for k in loss).backward()
            trainer.optimizer.step()
            #gpu  end time
            trainer.gpu_timer.toc()
            screen = [
                'Epoch %d/%d itr %d/%d:' % (epoch, cfg.end_epoch, itr, trainer.itr_per_epoch),
                'lr: %g' % (trainer.get_lr()),
                'speed: %.2f(%.2fs r%.2f)s/itr' % (
                    trainer.tot_timer.average_time, trainer.gpu_timer.average_time, trainer.read_timer.average_time),
                '%.2fh/epoch' % (trainer.tot_timer.average_time / 3600. * trainer.itr_per_epoch),
                ]
            screen += ['%s: %.4f' % ('loss_' + k, v.detach()) for k,v in loss.items()]
            #写入日志文件
            # trainer.logger.info(' '.join(screen))

            trainer.tot_timer.toc()
            trainer.tot_timer.tic()
            trainer.read_timer.tic()


        loss_file="../output/log/loss.txt"
        total_losse={key:value.item() for key,value in total_loss.items()}
        loss_str=json.dumps(total_losse)
        print(total_losse)
        with open(loss_file,"a") as f:            
            # f.truncate()  # 清空文件
            # if epoch == 0 :
            #   f.seek(0)
            f.write("epcoh:"+str(epoch)+" loss:"+loss_str+"\n")
        # save model: epoch model optimal
        trainer.save_model({
            'epoch': epoch,
            # state_dict是一个Python字典。保存了各层与其参数张量之间的映射
            'network': trainer.model.state_dict(),
            'optimizer': trainer.optimizer.state_dict(),
        }, epoch)
        

if __name__ == "__main__":
    main()
