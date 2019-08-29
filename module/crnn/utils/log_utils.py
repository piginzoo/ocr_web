#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 17-9-18 下午4:11
# @Author  : Luo Yao
# @Site    : http://github.com/TJCVRS
# @File    : log_utils.py
# @IDE: PyCharm Community Edition
"""
Set the log config
"""
import datetime
import logging
import os
import os.path as ops
from logging import handlers

import tensorflow as tf

FLAGS = tf.app.flags.FLAGS


def _p(tensor, msg):
    if (FLAGS.debug):
        dt = datetime.datetime.now().strftime('TF_DEBUG: %m-%d %H:%M:%S: ')
        msg = dt + msg
        return tf.Print(tensor, [tensor], msg, summarize=100)
    else:
        return tensor


def _p_shape(tensor, msg):
    if (FLAGS.debug):
        dt = datetime.datetime.now().strftime('TF_DEBUG: %m-%d %H:%M:%S: ')
        msg = dt + msg
        return tf.Print(tensor, [tf.shape(tensor)], msg, summarize=100)
    else:
        return tensor


def init_logger(level=logging.DEBUG,
                when="D",
                backup=7,
                _format="%(levelname)s: %(asctime)s: %(filename)s:%(lineno)d行 %(message)s"):
    log_path = ops.join(os.getcwd(), 'logs/shadownet.log')
    _dir = os.path.dirname(log_path)
    if not os.path.isdir(_dir):
        os.makedirs(_dir)

    logger = logging.getLogger()
    if not logger.handlers:
        formatter = logging.Formatter(_format)
        logger.setLevel(level)

        handler = handlers.TimedRotatingFileHandler(log_path, when=when, backupCount=backup)
        handler.setLevel(level)
        handler.setFormatter(formatter)
        logger.addHandler(handler)

        handler = logging.StreamHandler()
        handler.setLevel(level)
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger
