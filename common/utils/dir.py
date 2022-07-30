# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import os
import sys
#不存在则创建文件夹
def make_folder(folder_name):
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

def add_pypath(path):
    #sys.path:python搜索模块的路径列表
    #这种方法修改的sys.path作用域只是当前进程，进程结束后就失效了
    if path not in sys.path:
        sys.path.insert(0, path)

