# -*- coding: utf-8 -*-
"""
"""

import cv2
import numpy as np
import tensorflow as tf
from grpc.beta import implementations
from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2

tf.app.flags.DEFINE_string('IP', '127.0.0.1', '')
tf.app.flags.DEFINE_string('imgn', 'test.JPG', '')
FLAGS = tf.app.flags.FLAGS

print("FLAGS", FLAGS)


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
        pass

    def getCrnnRequest(self):
        return self.crnn

    def getCtpnnRequest(self):
        return self.ctpn


channels = Channels()
ctpn = channels.getCtpnnRequest()
print("ctpn", ctpn)
crnn = channels.getCrnnRequest()
print("crnn", crnn)


def test():
    imgName = FLAGS.imgn
    image = cv2.imread(imgName)
    image, _ = resize_image(image, 1200, 1600)
    # image = image[:, :, ::-1]
    print("image.shape", image.shape)
    h, w, c = image.shape
    im_info = np.array([h, w, c]).reshape([1, 3])
    print("im_info", im_info)
    image = np.array([image])

    ctpn["request"].inputs["input_image"].ParseFromString(
        tf.contrib.util.make_tensor_proto(image, dtype=tf.float32).SerializeToString())
    # ctpn["request"].inputs["input_im_info"].ParseFromString(
    #     tf.contrib.util.make_tensor_proto([], dtype=tf.float32).SerializeToString())
    response = ctpn["stub"].Predict(ctpn["request"], 60.0)
    print("response", response)


# 看哪个大了，就缩放哪个，规定大边的最大，和小边的最大
def resize_image(image, smaller_max, larger_max):
    h, w, _ = image.shape  # H,W

    # ----
    # |  |
    # |  |
    # |__|
    if h > w:
        if h < larger_max and w < smaller_max:
            return image, 1
        h_scale = larger_max / h
        w_scale = smaller_max / w
        # print("h_scale",h_scale,"w_scale",w_scale)
        scale = min(h_scale, w_scale)  # scale肯定是小于1的，越小说明缩放要厉害，所以谁更小，取谁
    # ___________
    # |         |
    # |_________|
    else:  # h<w
        if h < smaller_max and w < larger_max:
            return image, 1
        h_scale = smaller_max / h
        w_scale = larger_max / w
        scale = min(h_scale, w_scale)  # scale肯定是小于1的，越小说明缩放要厉害，所以谁更小，取谁

    # https://www.jianshu.com/p/11879a49d1a0 关于resize
    image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    cv2.imwrite("data/test.jpg", image)

    return image, scale


if __name__ == '__main__':
    print("imgn", FLAGS.imgn)
    print("IP", FLAGS.IP)
    test()
    pass
