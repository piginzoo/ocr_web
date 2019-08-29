# -*- coding: utf-8 -*-
"""
    说明：
"""

import logging

import cv2
import numpy as np
import os
import tensorflow as tf
from grpc.beta import implementations
from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2

from module.ctpn import ctpn_handle
from module.ctpn.utils.prepare import image_utils
from module.ctpn.utils.rpn_msr.proposal_layer import proposal_layer
from module.ctpn.utils.text_connector.detectors import TextDetector


def init_params():
    tf.app.flags.DEFINE_boolean('debug', True, '')
    tf.app.flags.DEFINE_boolean('evaluate', True, '')  # 是否进行评价（你可以光预测，也可以一边预测一边评价）
    tf.app.flags.DEFINE_boolean('split', True, '')  # 是否对小框做出评价，和画到图像上
    tf.app.flags.DEFINE_string('test_dir', '', '')  # 被预测的图片目录
    tf.app.flags.DEFINE_string('image_name', '', '')  # 被预测的图片名字，为空就预测目录下所有的文件
    tf.app.flags.DEFINE_string('pred_dir', 'data/pred', '')  # 预测后的结果的输出目录
    tf.app.flags.DEFINE_boolean('draw', True, '')  # 是否把gt和预测画到图片上保存下来，保存目录也是pred_dir
    tf.app.flags.DEFINE_boolean('save', True, '')  # 是否保存输出结果（大框、小框信息都要保存），保存到pred_dir目录里面去
    tf.app.flags.DEFINE_string('ctpn_model_dir', 'model/', '')  # model的存放目录，会自动加载最新的那个模型
    tf.app.flags.DEFINE_string('ctpn_model_file', '', '')  # 为了支持单独文件，如果为空，就预测test_dir中的所有文件

    tf.app.flags.DEFINE_string('test_images_dir', '', '')
    tf.app.flags.DEFINE_string('test_labels_dir', '', '')
    tf.app.flags.DEFINE_string('test_labels_split_dir', '', '')


tf.app.flags.DEFINE_string('IP', '127.0.0.1', '')
tf.app.flags.DEFINE_string('imgn', 'test.JPG', '')
FLAGS = tf.app.flags.FLAGS

logging.basicConfig(
    format='%(asctime)s - %(levelname)s : %(message)s',
    level=logging.DEBUG,
    handlers=[logging.StreamHandler()])
logger = logging.getLogger("WebServer")

logger.info("flags:%s", FLAGS)


class Channels:
    def __init__(self):
        def newChannel(name):
            channel = implementations.insecure_channel(FLAGS.IP, 8500)
            stub = prediction_service_pb2.beta_create_PredictionService_stub(channel)
            # 预测请求
            request = predict_pb2.PredictRequest()
            request.model_spec.name = name
            request.model_spec.signature_name = "serving_default"
            return stub, request

        predStub, predRequest = newChannel("crnn")
        self.crnn = {
            "stub": predStub,
            "request": predRequest
        }
        predStub, predRequest = newChannel("ctpn")
        self.ctpn = {
            "stub": predStub,
            "request": predRequest
        }
        logger.info("Channels init success")
        pass

    def getCrnnRequest(self):
        return self.crnn

    def getCtpnnRequest(self):
        return self.ctpn


channels = Channels()
ctpn = channels.getCtpnnRequest()
logger.info("ctpn channel %s", ctpn)
crnn = channels.getCrnnRequest()
logger.info("crnn channel %s", crnn)


def test():
    imgPath = FLAGS.imgn
    (_, image_name) = os.path.split(imgPath)
    original_img = cv2.imread(imgPath)
    image, scale = image_utils.resize_image(original_img, 1200, 1600)
    logger.info("image.shape:%s", image.shape)
    h, w, c = image.shape
    logger.debug('图像的h,w,c:%d,%d,%d', h, w, c)
    im_info = np.array([h, w, c]).reshape([1, 3])
    image = np.array([image])

    ctpn["request"].inputs["input_image"].ParseFromString(
        tf.contrib.util.make_tensor_proto(image, dtype=tf.float32).SerializeToString())
    logger.info("send predict request begin")
    response = ctpn["stub"].Predict(ctpn["request"], 60.0)
    logger.info("send predict request end")

    results = {}
    for key in response.outputs:
        tensor_proto = response.outputs[key]
        results[key] = tf.contrib.util.make_ndarray(tensor_proto)

    cls_prob = results["output_cls_prob"]
    bbox_pred = results["output_bbox_pred"]

    logger.debug("send predict request ===>>> results > cls_prob:%s", cls_prob)
    logger.debug("send predict request ===>>> results > bbox_pred:%s", bbox_pred)

    logger.info("start handel cls_prob")
    stat = ctpn_handle.cls_prob_val_reshape_debug_stat(cls_prob)
    logger.debug("前景返回概率情况:%s", stat)

    # 返回所有的base anchor调整后的小框，是矩形
    textsegs, _ = proposal_layer(cls_prob, bbox_pred, im_info)

    scores = textsegs[:, 0]
    textsegs = textsegs[:, 1:5]  # 这个是小框，是一个矩形 [1:5]=>1,2,3,4

    textdetector = TextDetector(DETECT_MODE='H')
    # 文本检测算法，用于把小框合并成一个4边型（不一定是矩形）
    boxes = textdetector.detect(textsegs, scores[:, np.newaxis], image.shape[:2])
    # box是9个值，4个点，8个值了吧，还有个置信度：全部小框得分的均值作为文本行的均值
    boxes = np.array(boxes, dtype=np.int)
    # boxes, scores, textsegs

    # scale 放大 unresize back回去
    boxes_big = np.array(image_utils.resize_labels(boxes[:, :8], 1 / scale))
    bbox_small = np.array(image_utils.resize_labels(textsegs, 1 / scale))

    _image = {'name': image_name, 'boxes': boxes_big}

    draw_image, f1 = ctpn_handle.post_detect(bbox_small, boxes_big, image_name, original_img, scores)
    if draw_image is not None: _image['image'] = draw_image
    if draw_image is not None: _image['f1'] = f1
    logger.info("ctpn end by result:%s", _image)


if __name__ == '__main__':
    logger.info("IP:%s, imgn:%s", FLAGS.IP, FLAGS.imgn)
    init_params()
    test()
    pass
