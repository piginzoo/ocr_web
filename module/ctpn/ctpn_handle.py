# -*- coding: utf-8 -*-
"""
    说明：
"""

from module.ctpn.utils import stat
from module.ctpn.utils.rpn_msr.config import Config


def cls_prob_val_reshape_debug_stat(cls_prob_val):
    cls_prob_for_debug = cls_prob_val.reshape(1,
                                              cls_prob_val.shape[1],  # H
                                              cls_prob_val.shape[2] // Config.NETWORK_ANCHOR_NUM,  # W
                                              Config.NETWORK_ANCHOR_NUM,  # 每个点上扩展的10个anchors
                                              -1)  # <---2个值, 0:背景概率 1:前景概率
    _stat = stat(cls_prob_for_debug[:, :, :, :, 1].reshape(-1, 1))  # 去掉背景，只保留前景，然后传入统计
    return _stat
