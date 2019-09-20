# -*- coding: utf-8 -*-
"""
    说明：
"""
from server import conf
import grpc
import logging
import tensorflow as tf
import numpy as np
from tensorflow.contrib.util import make_tensor_proto
from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2_grpc

logger = logging.getLogger(__name__)

def create_channel(name, IP, PORT):
    logger.info("TF Serving 通道连接 - name:%s IP:%s PORT:%s", name, IP, PORT)
    channel = grpc.insecure_channel("{}:{}".format(IP, PORT))
    stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)

    # 预测请求
    request = predict_pb2.PredictRequest()
    request.model_spec.name = name
    request.model_spec.signature_name = tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY
    logger.info("链接模型[%s]的通道创建,IP:%s,端口:%d,", name, IP, PORT)

    return stub, request


def crnn_tf_serving_call(_input_data,_batch_size_array):

    # 测试用代码，不用连接tf-serving docker服务
    inputdata = tf.placeholder(dtype=tf.float32,shape=[64,128,3682],name='input')
    greedy_decodes, greedy_prob = tf.nn.ctc_greedy_decoder(inputs=inputdata,sequence_length=np.array(128*[50]),merge_repeated=True)
    sess = tf.Session()
    with sess.as_default():
        greedy_d,_ = sess.run([greedy_decodes, greedy_prob],feed_dict={inputdata: _input_data})
    _decodes=greedy_d[0]
    return _decodes.indices, _decodes._decodes, _decodes.dense_shape

    stub, request = create_channel(conf.CRNN_NAME, conf.TF_SERVING_IP, conf.TF_SERVING_PORT)

    request.inputs["input_data"].CopyFrom(make_tensor_proto(np.array(_input_data), dtype=tf.float32))
    request.inputs["input_batch_size"].CopyFrom(make_tensor_proto(_batch_size_array))

    logger.debug("调用CRNN模型预测，开始：调用TF-Server，IP：%s,端口：%d",conf.TF_SERVING_IP, conf.TF_SERVING_PORT)
    response = stub.Predict(request, 60.0)
    logger.debug("调用CRNN模型预测，结束")

    results = {}
    for key in response.outputs:
        logger.debug("CRNN模型返回参数：%r", key)
        tensor_proto = response.outputs[key]
        results[key] = tf.contrib.util.make_ndarray(tensor_proto)

    # output_net_out_index = results["output_net_out_index"]
    # B(output_net_out_index)
    # logger.info("output_net_out_index:%s", output_net_out_index)
    # logger.info("output_net_out_index.shape:%s", output_net_out_index.shape)
    # logger.debug("output_indices.shape:%s", output_indices.shape)
    # logger.debug("output_shape.shape:%s", output_shape.shape)
    # logger.debug("output_values.shape:%s", output_values.shape)
    # preds_sparse = tf.SparseTensor(output_indices, output_values, output_shape)

    # A.解决单个SparseTensor无法被识别的问题，红岩之前的解决方案

    output_indices = results["output_indices"]
    output_values = results["output_values"]
    output_shape = results["output_shape"]

    return output_indices,output_values,output_shape


def ctpn_tf_serving_call(image,im_info):
    # 测试用代码，不用连接tf-serving docker服务
    return None,None

    stub, request = create_channel(conf.CRNN_NAME, conf.TF_SERVING_IP, conf.TF_SERVING_PORT)

    request.inputs["input_image"].CopyFrom(make_tensor_proto(np.array([image]),dtype=tf.float32))
    request.inputs["input_im_info"].CopyFrom(make_tensor_proto(np.array([im_info]),dtype=tf.float32))
    logger.debug("调用CTPN模型预测，开始")
    response = stub.Predict(request, 60.0)
    logger.debug("调用CTPN模型预测，结束")

    results = {}
    for key in response.outputs:
        tensor_proto = response.outputs[key]
        results[key] = tf.contrib.util.make_ndarray(tensor_proto)

    cls_prob = results["output_cls_prob"]
    bbox_pred = results["output_bbox_pred"]
    return cls_prob,bbox_pred

