# -*- coding: utf-8 -*-
# @Author : zheng
# @Date : 2022/8/18 10:13

import os
import sys
import datetime
import logging


def build_logger(name=None, log_dir=None, level=logging.INFO, mode='w',
                fmt='[%(asctime)s] [%(name)s] [%(levelname)s] - %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S', type='train') -> logging.Logger:
    if not name:
        name = __name__
    rootLogger = logging.getLogger(name)
    rootLogger.propagate = False

    consoleHandler = logging.StreamHandler(sys.stdout)  #################>??????????//
    consoleHandler.setFormatter(logging.Formatter(fmt=fmt, datefmt=datefmt))
    attached_to_std = False
    for handler in rootLogger.handlers:
        if isinstance(handler, logging.StreamHandler):
            if handler.stream == sys.stderr or handler.stream == sys.stdout:
                attached_to_std = True
                break
    if not attached_to_std:
        rootLogger.addHandler(consoleHandler)
    rootLogger.setLevel(level)
    consoleHandler.setLevel(level)

    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
        log_path = os.path.join(log_dir, '{}.log'.format(type))
        fileHandler = logging.FileHandler(log_path, mode=mode)
        fileHandler.setFormatter(logging.Formatter(fmt, datefmt=datefmt))
        rootLogger.addHandler(fileHandler)
        fileHandler.setLevel(level)

    return rootLogger

