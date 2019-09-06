# -*- coding: utf-8 -*-
"""
    说明：
"""

import logging
import os

import cv2
import tensorflow as tf

from config import param_config
import ocr_utils
from module.crnn import crnn
from module.ctpn import ctpn

FLAGS = tf.app.flags.FLAGS

logging.basicConfig(
    format='%(asctime)s - %(levelname)s : %(message)s',
    level=logging.DEBUG,
    handlers=[logging.StreamHandler()])

logger = logging.getLogger("WebServer")

logger.info("flags:%s", FLAGS)


def pridict():
    imgPath = FLAGS.imgn
    (_, image_name) = os.path.split(imgPath)
    original_img = cv2.imread(imgPath)
    # ctpn_predict
    result = ctpn.ctpn_predict(original_img, image_name)
    result_image = result[0]['boxes']
    small_images = ocr_utils.crop_small_images(original_img, result_image)
    # crnn_predict
    crnn_result = crnn.crnn_predict(small_images, param_config.CRNN_BATCH_SIZE)
    result[0]['text'] = crnn_result
    return result[0]


if __name__ == '__main__':
    logger.info("IP:%s, imgn:%s", FLAGS.IP, FLAGS.imgn)
    result = pridict()
    logger.debug("最终预测结果：%s", result)
    pass
