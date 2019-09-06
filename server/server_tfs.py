# -*- coding:utf-8 -*-
import base64
import logging
import os
import time
from threading import current_thread

import cv2
import numpy as np
import tensorflow as tf
from flask import Flask, jsonify, request, abort, render_template, Response

import api
import ocr_utils
from config import param_config
from module.crnn import crnn
from module.ctpn import ctpn

DEBUG = False
FLAGS = tf.app.flags.FLAGS

logging.basicConfig(
    format='%(asctime)s : %(levelname)s : %(message)s',
    level=logging.INFO,
    handlers=[logging.StreamHandler()])

logger = logging.getLogger("WebServer")

logger.debug('子进程:%s,父进程:%s,线程:%r', os.getpid(), os.getppid(), current_thread())

cwd = os.getcwd()
app = Flask(__name__, root_path="web")
app.jinja_env.globals.update(zip=zip)


def pridict(original_img, image_name="test.jpg", is_verbose=False):
    """
    预测图片
    :param original_img: image
    :param image_name:  name
    :return:
    """
    # imgPath = FLAGS.imgn
    # (_, image_name) = os.path.split(imgPath)
    # original_img = cv2.imread(imgPath)
    # ctpn_predict
    result = ctpn.ctpn_predict(original_img, image_name)
    result_image = result[0]['boxes']
    small_images = ocr_utils.crop_small_images(original_img, result_image)
    # crnn_predict
    crnn_result = crnn.crnn_predict(small_images, param_config.CRNN_BATCH_SIZE)
    result[0]['text'] = crnn_result
    if is_verbose:
        # 小框们的图片的base64
        result[0]['small_images'] = ocr_utils.nparray2base64(small_images)
        # 这个是为了，把图片再回显到网页上用
        for r in result:
            # 从opencv的np array格式，转成原始图像，再转成base64
            if r.get('image', None) is not None:
                logger.debug("返回的大图:%r", r['image'].shape)
                r['image'] = ocr_utils.nparray2base64(r['image'])
            else:
                logger.debug("返回大图为空")
        # logger.debug("最终的预测的子图:%r",result[0]['small_images'])

    return True, result[0]


def decode2img(buffer):
    logger.debug("从web读取数据len:%r", len(buffer))

    if len(buffer) == 0: return False, "图片是空"

    # 先给他转成ndarray(numpy的)
    data_array = np.frombuffer(buffer, dtype=np.uint8)

    # 从ndarray中读取图片，有raw数据变成一个图片GBR数据,出来的数据，其实就是有维度了，就是原图的尺寸，如160x70
    image = cv2.imdecode(data_array, cv2.IMREAD_COLOR)

    if image is None:
        logger.error("图像解析失败")  # 有可能从字节数组解析成图片失败
        return None

    logger.debug("从字节数组变成图像的shape:%r", image.shape)

    return image


@app.route("/")
def index():
    # with open("../version") as f:
    #     version = f.read()
    logger.info("index time:%s", time.time())
    return render_template('index.html', version="version")


# base64编码的小图片的识别，这个制作OCR文字识别，不做文字弹出了
@app.route('/crnn', methods=['POST'])
def do_crnn():
    result = None
    try:
        param_config.disable_debug_flags()  # 不用处理调试的动作，但是对post方式，还是保留
        buffers = api.process_crnn_request(request)
        images = []
        for b in buffers:
            image = decode2img(b)
            images.append(image)
        # crnn_predict
        result = crnn.crnn_predict(images, param_config.CRNN_BATCH_SIZE)  # 小框们的文本们
        result = api.post_crnn_process(result)
    except Exception as e:
        import traceback
        traceback.print_exc()
        logger.error("处理图片过程中出现问题：%r", e)
        return jsonify({'error_code': -1, 'message': str(e)})

    if result is None:
        return jsonify({'error_code': -1, 'message': 'image resolve fail'})
    else:
        return jsonify(result)


# base64编码的图片识别
@app.route('/ocr', methods=['POST'])
def ocr_base64():
    logger.debug("post calling...")

    try:
        param_config.disable_debug_flags()  # 不用处理调试的动作，但是对post方式，还是保留
        buffer = api.process_request(request)
        image = decode2img(buffer)
        height, width, _ = image.shape
        if image is None:
            return jsonify({'error_code': -1, 'message': 'image decode from base64 failed.'})

        if DEBUG:
            success = True
            result = {  # 测试用
                'name': 'xxx.png',
                'boxes': [[1, 1, 1, 1],
                          [2, 2, 2, 2]],
                'text': ['xxxxxxx', 'yyyyyyy']
            }
        else:
            success, result = pridict(image)

        if success:
            result = api.post_process(result, width, height)
    except Exception as e:
        import traceback
        traceback.print_exc()
        logger.error("处理图片过程中出现问题：%r", e)
        return jsonify({'error_code': -1, 'message': str(e)})

    if success:
        if result is None:
            return jsonify({'error_code': -1, 'message': 'image resolve fail'})
        else:
            return jsonify(result)
    else:
        return jsonify({'error_code': -1, 'message': result})


# 图片的识别
@app.route('/ocr.post', methods=['POST'])
def ocr():
    data = request.files['image']
    image_name = data.filename
    buffer = data.read()
    image = decode2img(buffer)
    if image is None:
        abort(500)
        abort(Response('解析Web传入的图片失败'))

    logger.info("获得上传图片[%s]，尺寸：%d 字节", image_name, len(image))
    start = time.time()
    success, result = pridict(image, image_name, is_verbose=True)
    logger.info("识别图片[%s]花费[%d]秒", image_name, time.time() - start)
    return render_template('result.html', result=result)


# 图片的识别
@app.route('/test', methods=['GET'])
def test():
    with open("test/test.png", "rb") as f:
        data = base64.b64encode(f.read())
        data = str(data, 'utf-8')
    aa = ['a2', 'a1']
    bb = ['b2', 'b1']
    return render_template('test.html', data=data, aa=aa, bb=bb)


def startup():
    pass
    # # 测试代码
    # with open("test/test.png","rb") as f:
    #     image = f.read()
    # process(image,"test.jpg")


if __name__ == "__main__":
    startup()
