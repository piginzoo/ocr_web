#-*- coding:utf-8 -*- 
from flask import Flask,jsonify,request,abort,render_template,Response
import base64,cv2, numpy as np
import logging
from server import conf

from threading import current_thread
import time
from utils import ocr_utils, api
from ctpn.main import pred as ctpn
from crnn.tools import pred as crnn
import os
import utils.tf_serving_agent as tfserving

logging.basicConfig(
    format='%(asctime)s : %(levelname)s : %(message)s',
    level=logging.DEBUG,
    handlers=[logging.StreamHandler()])

logger = logging.getLogger("WebServer")

logger.debug('子进程:%s,父进程:%s,线程:%r', os.getpid(), os.getppid(), current_thread())

mode = conf.init_arguments()

ctpn_params = None
crnn_params = None

if mode=="single":
    import tensorflow as tf
    logger.info("启动加载模型模式")
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.3, allow_growth=True)
    config = tf.ConfigProto(gpu_options=gpu_options)
    config.allow_soft_placement = True
    logger.debug("开始初始化CTPN")
    ctpn_params = ctpn.initialize(config)
    logger.debug("开始初始化CRNN")
    crnn_params = crnn.initialize(config)

if mode=="tfserving":
    logger.info("启动TF-Serving模式")

cwd = os.getcwd()
app = Flask(__name__,root_path="web")
app.jinja_env.globals.update(zip=zip)


# 参考：https://www.cnblogs.com/haolujun/p/9778939.html
# gc.freeze() #调用gc.freeze()必须在fork子进程之前，在gunicorn的这个地方调用正好合适，freeze把截止到当前的所有对象放入持久化区域，不进行回收，从而model占用的内存不会被copy-on-write。

def decode2img(buffer):
    logger.debug("从web读取数据len:%r",len(buffer))

    if len(buffer)==0: return False,"图片是空"

    # 先给他转成ndarray(numpy的)
    data_array = np.frombuffer(buffer,dtype=np.uint8)

    # 从ndarray中读取图片，有raw数据变成一个图片GBR数据,出来的数据，其实就是有维度了，就是原图的尺寸，如160x70
    image = cv2.imdecode(data_array, cv2.IMREAD_COLOR)

    if image is None:
        logger.error("图像解析失败")#有可能从字节数组解析成图片失败
        return None

    logger.debug("从字节数组变成图像的shape:%r",image.shape)

    return image

#读入的buffer是个纯byte数据
def process(image,image_name="test.jpg",is_verbose=False):

    # result:[{
    #     name: 'xxx.png',
    #     'box': {
    #         [1, 1, 1, 1],
    #         [2, 2, 2, 2]
    #     },
    #     image : <draw image numpy array>,
    #     'f1': 0.78
    #     }
    # }, ]
    print(mode)
    call_back = None if mode=="single" else tfserving.ctpn_tf_serving_call
    print(call_back)
    result = ctpn.pred(ctpn_params,[image],[image_name],call_back)

    # logger.debug("预测返回结果：%r",result[0])
    small_images = ocr_utils.crop_small_images(image, result[0]['boxes'])
    result[0]['text'] = process_crnn(small_images)    # 小框们的文本们

    # 仅仅调试CTPN
    # all_txt = ['']*len(small_images)

    # 返回原图和切出来的小图，这个是为了调试用
    if is_verbose:
        # 小框们的图片的base64
        result[0]['small_images'] = ocr_utils.nparray2base64(small_images)
        # 这个是为了，把图片再回显到网页上用
        for r in result:
            # 从opencv的np array格式，转成原始图像，再转成base64
            if r.get('image',None) is not None:
                logger.debug("返回的大图:%r",r['image'].shape)
                r['image'] = ocr_utils.nparray2base64(r['image'])
            else:
                logger.debug("返回大图为空")
        # logger.debug("最终的预测的子图:%r",result[0]['small_images'])

    return True,result[0]


def process_crnn(small_images):
    call_back = None if mode == "single" else tfserving.crnn_tf_serving_call
    all_txt,_ = crnn.pred(crnn_params,small_images, conf.CRNN_BATCH_SIZE,call_back)
    logger.debug("最终的预测结果为：%r",all_txt)
    return all_txt

@app.route("/")
def index():
    # with open("../version") as f:
    #     version = f.read()
    return render_template('index.html',version="version")


# base64编码的小图片的识别，这个制作OCR文字识别，不做文字弹出了
@app.route('/crnn',methods=['POST'])
def do_crnn():
    try:
        if mode == "single": conf.disable_debug_flags() # 不用处理调试的动作，但是对post方式，还是保留
        buffers = api.process_crnn_request(request)
        images = []
        for b in buffers:
            image = decode2img(b)
            images.append(image)
        result = process_crnn(images)  # 小框们的文本们
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
@app.route('/ocr',methods=['POST'])
def ocr_base64():

    logger.debug("post calling...")

    try:
        if mode=="single": conf.disable_debug_flags() # 不用处理调试的动作，但是对post方式，还是保留
        buffer = api.process_request(request)

        image = decode2img(buffer)
        height,width,_ = image.shape
        if image is None:
            return jsonify({'error_code':-1,'message':'image decode from base64 failed.'})

        success,result = process(image)

        if result:
            result = api.post_process(result, width, height)
    except Exception as e:
        import traceback
        traceback.print_exc()
        logger.error("处理图片过程中出现问题：%r",e)
        return jsonify({'error_code':-1,'message':str(e)})
    
    if success: 
        if result is None:
            return jsonify({'error_code':-1,'message':'image resolve fail'})
        else:
            return jsonify(result)
    else:
        return jsonify({'error_code':-1,'message':result})


# 图片的识别
@app.route('/ocr.post',methods=['POST'])
def ocr():
    data = request.files['image']
    image_name = data.filename
    buffer = data.read()
    image = decode2img(buffer)
    if image is None:
        abort(500)
        abort(Response('解析Web传入的图片失败'))

    logger.debug("获得上传图片[%s]，尺寸：%d 字节", image_name,len(image))
    start = time.time()
    success,result = process(image,image_name,is_verbose=True)
    logger.debug("识别图片[%s]花费[%d]秒",image_name,time.time()-start)
    return render_template('result.html', result=result)


# 图片的识别
@app.route('/test',methods=['GET'])
def test():
    with open("test/test.png", "rb") as f:
        data = base64.b64encode(f.read())
        data = str(data, 'utf-8')
    aa = ['a2','a1']
    bb = ['b2','b1']
    return render_template('test.html', data=data,aa=aa,bb=bb)


def startup():
    pass

    # # 测试代码
    # with open("test/test.png","rb") as f:
    #     image = f.read()
    # process(image,"test.jpg")

if __name__=="__main__":

    startup()

