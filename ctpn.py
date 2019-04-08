import sys,logging
sys.path.append("../ctpn")

import main.pred as ctpn
import tensorflow as tf

tf.app.flags.DEFINE_boolean('debug_mode', True, '')
tf.app.flags.DEFINE_boolean('evaluate', True, '') # 是否进行评价（你可以光预测，也可以一边预测一边评价）
tf.app.flags.DEFINE_boolean('split', True, '')    # 是否对小框做出评价，和画到图像上
tf.app.flags.DEFINE_string('file', '', '')     # 为了支持单独文件，如果为空，就预测test_home中的所有文件
tf.app.flags.DEFINE_string('gpu', '0', '')
tf.app.flags.DEFINE_boolean('draw', True, '') # 是否把gt和预测画到图片上保存下来，保存目录也是pred_home
tf.app.flags.DEFINE_boolean('save', True, '') # 是否保存输出结果（大框、小框信息都要保存），保存到pred_home目录里面去
tf.app.flags.DEFINE_string('model', '../ctpn/model/', '') # model的存放目录，会自动加载最新的那个模型

logging.basicConfig(
    format='%(asctime)s : %(levelname)s : %(message)s',
    level=logging.DEBUG,
    handlers=[logging.StreamHandler()])

ctpn.pred([])

