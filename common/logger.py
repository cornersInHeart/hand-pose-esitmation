# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import logging
import os

OK = '\033[92m'
WARNING = '\033[93m'
FAIL = '\033[91m'
END = '\033[0m'

PINK = '\033[95m'
BLUE = '\033[94m'
GREEN = OK
RED = FAIL
WHITE = END
YELLOW = WARNING

class colorlogger():
    def __init__(self, log_dir, log_name='train_logs.txt'):
        # 1.set logger
        self._logger = logging.getLogger(log_name)
        #设置log级别
        self._logger.setLevel(logging.INFO)
        #日志文件的路径
        log_file = os.path.join(log_dir, log_name)
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        # 2.create handler，用于写入日志文件
        file_log = logging.FileHandler(log_file, mode='a')
        # 输出到file的log等级的开关
        file_log.setLevel(logging.INFO)
        # create handler,输出到控制台
        console_log = logging.StreamHandler()
        # 输出到console的log等级
        console_log.setLevel(logging.INFO)
        # 3.定义handler的输出格式
        formatter = logging.Formatter(
            "{}%(asctime)s{} %(message)s".format(GREEN, END),
            "%m-%d %H:%M:%S")
        file_log.setFormatter(formatter)
        console_log.setFormatter(formatter)
        # 4.将logger添加到handler里面
        self._logger.addHandler(file_log)
        self._logger.addHandler(console_log)

    def debug(self, msg):
        self._logger.debug(str(msg))

    def info(self, msg):
        self._logger.info(str(msg))

    def warning(self, msg):
        self._logger.warning(WARNING + 'WRN: ' + str(msg) + END)

    def critical(self, msg):
        self._logger.critical(RED + 'CRI: ' + str(msg) + END)

    def error(self, msg):
        self._logger.error(RED + 'ERR: ' + str(msg) + END)

