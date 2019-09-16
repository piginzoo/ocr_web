# -*- coding: utf-8 -*-
"""
    说明：
"""
import logging
import os
import numpy as np
import tensorflow as tf
from tensorflow.contrib.util import make_tensor_proto
import time
import conf
import module.crnn.utils.image_util as image_util
import module.tfs.channels as channel
from module.crnn.config import config
from module.crnn.utils import data_utils

logger = logging.getLogger("crnn")
FLAGS = tf.app.flags.FLAGS

charset = data_utils.get_charset(os.path.join(os.path.abspath(''), "module/crnn/charset.3770.txt"))


# crnn predict
def crnn_predict(image_list, _batch_size):
    start_time = time.time()
    pred_result = []  # 预测结果，每个元素是一个字符串
    for i in range(0, len(image_list), _batch_size):
        # 计算实际的batch大小，最后一批数量可能会少一些
        begin = i
        end = i + _batch_size
        if begin + _batch_size > len(image_list):
            end = len(image_list)
        count = end - begin

        logger.debug("从所有图像[%d]抽取批次，从%d=>%d", len(image_list), begin, end)
        _input_data = image_list[begin:end]
        # logger.debug("抽取批次结果：%s", _input_data)
        _input_data = image_util.resize_batch_image(_input_data, config.INPUT_SIZE, FLAGS.resize_mode)
        # logger.debug("_input_data:%s", _input_data)

        # batch_size，也就是CTC的sequence_length数组要求的格式是：
        # 长度是batch个，数组每个元素是sequence长度，也就是64个像素 [64,64,...64]一共batch个。
        _batch_size_array = np.array(count * [config.SEQ_LENGTH]).astype(np.int32)
        # logger.debug("_batch_size_array:%s", _batch_size_array)

        stub , request = channel.create_channel(conf.CRNN_NAME,conf.TF_SERVING_IP,conf.TF_SERVING_PORT)

        request.inputs["input_data"].CopyFrom(make_tensor_proto(np.array(_input_data),dtype=tf.float32))
        request.inputs["input_batch_size"].CopyFrom(make_tensor_proto(_batch_size_array))

        logger.debug("调用CRNN模型预测，开始")
        response = stub.Predict(request, 60.0)
        logger.debug("调用CRNN模型预测，结束")

        results = {}
        for key in response.outputs:
            tensor_proto = response.outputs[key]
            results[key] = tf.contrib.util.make_ndarray(tensor_proto)

        # output_net_out_index = results["output_net_out_index"]
        # output_shape = results["output_shape"]
        # output_indices = results["output_indices"]
        # output_values = results["output_values"]
        # B(output_net_out_index)
        # logger.info("output_net_out_index:%s", output_net_out_index)
        # logger.info("output_net_out_index.shape:%s", output_net_out_index.shape)
        # logger.debug("output_indices.shape:%s", output_indices.shape)
        # logger.debug("output_shape.shape:%s", output_shape.shape)
        # logger.debug("output_values.shape:%s", output_values.shape)
        # preds_sparse = tf.SparseTensor(output_indices, output_values, output_shape)
        preds_sparse = results["output"]
        preds = data_utils.sparse_tensor_to_str_new(preds_sparse, charset)
        pred_result += preds
    logger.debug("CRNN预测结果:%s", pred_result)
    logger.info("全部CRNN预测完成，共用时：%s", (time.time() - start_time))
    return pred_result


def B(values):
    """
    1 state
    2 ssta atee
    3 s state
    4 ss s ta te
    5 -s-s-st-a-t-e
    字符说明：
        -（减号）表示空格
        " "(空格)表示两个同样的字符不去重
        其它全都原样显示
    :param values: 是一个二维数组
    :return:
    """
    res = []
    for data in values:
        row = []
        rd = None
        # print(data)
        for idx in data:
            char = charset[idx]
            if char == '-':
                continue
            elif char == ' ':
                rd = None
                continue

            if char == rd:
                continue
            else:
                row.append(char)
            rd = char
        if len(row) > 0:
            logger.info("row: %s   =========  %s", "".join(row), row)
            res.append(row)
