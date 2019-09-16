# -*- coding: utf-8 -*-
"""
    说明：
"""
import logging

import grpc
from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2_grpc

logger = logging.getLogger(__name__)


def create_channel(name, IP, PORT):
    logger.info("TF Serving 通道连接 - name:%s IP:%s PORT:%s", name, IP, PORT)
    channel = grpc.insecure_channel(IP, PORT)
    stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)

    # 预测请求
    request = predict_pb2.PredictRequest()
    request.model_spec.name = name
    request.model_spec.signature_name = "ocr serving"
    logger.info("链接模型[%s]的通道创建,IP:%s,端口:%d,", name, IP, PORT)

    return stub, request



