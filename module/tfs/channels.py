# -*- coding: utf-8 -*-
"""
    说明：
"""
import logging

import tensorflow as tf
from grpc.beta import implementations
from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2

from config import config

logger = logging.getLogger("channels")
FLAGS = tf.app.flags.FLAGS


class __Channels:
    def __init__(self):
        gRPC = config.get("gRPC")
        IP = gRPC.get("IP")
        PORT = gRPC.get("PORT")
        def newChannel(name):
            channel = implementations.insecure_channel(IP, PORT)
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


global ctpn, crnn


def init():
    global ctpn, crnn
    channels = __Channels()
    ctpn = channels.getCtpnnRequest()
    logger.info("ctpn init channel %s", ctpn)
    crnn = channels.getCrnnRequest()
    logger.info("crnn init channel %s", crnn)
