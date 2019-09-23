# -*- coding: utf-8 -*-
"""
    说明：
"""
import logging

import numpy as np
import tensorflow as tf
from tensorflow.contrib.util import make_tensor_proto
import time
from server import conf

import utils.channels as channel
from utils.ctpn import ctpn_handle
from utils.ctpn.utils.prepare import image_utils
from utils.ctpn.utils.rpn_msr.proposal_layer import proposal_layer
from utils.ctpn.utils.text_connector.detectors import TextDetector

logger = logging.getLogger("ctpn")


# ctpn predict
# def ctpn_predict(original_img, image_name):
#     start_time = time.time()
#     image, scale = image_utils.resize_image(original_img, 1200, 1600)
#
#     logger.info("image.shape:%s", image.shape)
#     h, w, c = image.shape
#     logger.debug('图像的h,w,c:%d,%d,%d', h, w, c)
#     im_info = np.array([h, w, c]).reshape([1, 3])
#     image = image[:, :, ::-1]
#     # image = np.array([image])
#
#     stub, request = channel.create_channel(conf.CTPN_NAME, conf.TF_SERVING_IP, conf.TF_SERVING_PORT)
#
#     request.inputs["input_image"].CopyFrom(make_tensor_proto(np.array([image]),dtype=tf.float32))
#     request.inputs["input_im_info"].CopyFrom(make_tensor_proto(np.array([im_info]),dtype=tf.float32))
#     logger.debug("调用CTPN模型预测，开始")
#     response = stub.Predict(request, 60.0)
#     logger.debug("调用CTPN模型预测，结束")
#
#     results = {}
#     for key in response.outputs:
#         tensor_proto = response.outputs[key]
#         results[key] = tf.contrib.util.make_ndarray(tensor_proto)
#
#     cls_prob = results["output_cls_prob"]
#     bbox_pred = results["output_bbox_pred"]
#
#     # logger.debug("send predict request ===>>> results > cls_prob:%s", cls_prob)
#     # logger.debug("send predict request ===>>> results > bbox_pred:%s , shape:%s", bbox_pred[0][0][0], bbox_pred.shape)
#     logger.info("ctpn start handel cls_prob,bbox_pred")
#     stat = ctpn_handle.cls_prob_val_reshape_debug_stat(cls_prob)
#     logger.debug("前景返回概率情况:%s", stat)
#
#     # 返回所有的base anchor调整后的小框，是矩形
#     textsegs, _ = proposal_layer(cls_prob, bbox_pred, im_info)
#
#     scores = textsegs[:, 0]
#     textsegs = textsegs[:, 1:5]  # 这个是小框，是一个矩形 [1:5]=>1,2,3,4
#
#     textdetector = TextDetector(DETECT_MODE='H')
#     # 文本检测算法，用于把小框合并成一个4边型（不一定是矩形）
#     boxes = textdetector.detect(textsegs, scores[:, np.newaxis], image.shape[:2])
#     # box是9个值，4个点，8个值了吧，还有个置信度：全部小框得分的均值作为文本行的均值
#     boxes = np.array(boxes, dtype=np.int)
#     # logger.debug("results > len:%s boxes:%s , shape:%s", len(boxes), boxes, boxes.shape)
#     # logger.debug("results > len:%s boxes:%s , shape:%s", len(boxes[0]), boxes[0], boxes.shape)
#     # boxes, scores, textsegs
#
#     # scale 放大 unresize back回去
#     boxes_big = np.array(image_utils.resize_labels(boxes[:, :8], 1 / scale))
#     bbox_small = np.array(image_utils.resize_labels(textsegs, 1 / scale))
#
#     _image = {'name': image_name, 'boxes': boxes_big}
#
#     draw_image, f1 = ctpn_handle.post_detect(bbox_small, boxes_big, image_name, original_img, scores)
#     if draw_image is not None:
#         logger.info("draw_image is not None")
#         _image['image'] = draw_image
#         _image['f1'] = f1
#     result = []
#     result.append(_image)
#     logger.info("CTPN总共用时：%s", (time.time() - start_time))
#     return result