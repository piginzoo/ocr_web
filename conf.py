################################################################
#
#   定义CTPN、CRNN的共同参数
#
################################################################

# 定义相关的目录，这个是为了方便3个项目集成，所以需要绝对路径，方便import，不是为了找文件，找文件用相对路径就好，是为了python import
CTPN_HOME=['/Users/piginzoo/workspace/opensource/ctpn','/app.fast/projects/ctpn']

# CTPN常用参数
CTPN_MODEL_FILE= "ctpn_50000.ckpt" # 定义模型目录，如果FILE被定义，直接加载FILE，否则，挑选最新的模型加载(即在checkpoint文件中记录的）
CTPN_MODEL_FILE= "ctpn-2019-04-24-17-50-18-10000.ckpt"

CTPN_MODEL_DIR= "../ctpn/model"
CTPN_DRAW = True
CTPN_SAVE = True        # 是否保存识别的坐标
CTPN_EVALUATE = True    # 是否提供评估
CTPN_SPLIT = False      # 是否保留小框(CTPN网络识别结果）
CTPN_PRED_DIR = "data/pred" # 保存的内容存放的目录
CTPN_TEST_DIR = "data/test" #

# CRNN常用参数
CRNN_HOME=['/Users/piginzoo/workspace/opensource/crnn','/app.fast/projects/crnn']
CRNN_MODEL_FILE= "crnn_2019-04-30-17-17-21.ckpt-0"
CRNN_MODEL_DIR="../crnn/model"
CRNN_CHARSET_FILE="../crnn/charset6k.txt"
CRNN_BATCH_SIZE=32

# 通用的调试开关
DEBUG=True


# 定义各类参数
def init_arguments():
    import tensorflow as tf

    # 共享的
    tf.app.flags.DEFINE_boolean('debug',DEBUG, '')      # 是否调试

    # 这个是为了兼容
    # gunicorn -w 2 -k gevent web.api_server:app -b 0.0.0.0:8080
    tf.app.flags.DEFINE_string('worker-class', 'gevent', '')
    tf.app.flags.DEFINE_integer('workers', 2, '')
    tf.app.flags.DEFINE_string('bind', '0.0.0.0:8080', '')
    tf.app.flags.DEFINE_integer('timeout', 60, '')

    # ctpn的
    tf.app.flags.DEFINE_string('ctpn_model_dir', CTPN_MODEL_DIR, '') # model的存放目录，会自动加载最新的那个模型
    tf.app.flags.DEFINE_string('ctpn_model_file',CTPN_MODEL_FILE, '')# 为了支持单独文件，如果为空，就挑选最新的model
    tf.app.flags.DEFINE_boolean('evaluate', CTPN_EVALUATE, '')  # 是否进行评价（你可以光预测，也可以一边预测一边评价）
    tf.app.flags.DEFINE_boolean('split',    CTPN_SPLIT, '')     # 是否对小框做出评价，和画到图像上
    tf.app.flags.DEFINE_boolean('draw',     CTPN_DRAW, '')      # 是否把gt和预测画到图片上保存下来，保存目录也是pred_home
    tf.app.flags.DEFINE_boolean('save',     CTPN_SAVE, '')      # 是否保存输出结果（大框、小框信息都要保存），保存到pred_home目录里面去
    tf.app.flags.DEFINE_string('pred_dir', CTPN_PRED_DIR, '')  # 预测后的结果的输出目录
    tf.app.flags.DEFINE_string('test_dir', CTPN_TEST_DIR, '')  # 预测后的结果的输出目录

    # crnn的
    tf.app.flags.DEFINE_string('crnn_model_dir', CRNN_MODEL_DIR, '')
    tf.app.flags.DEFINE_string('crnn_model_file',CRNN_MODEL_FILE, '')
    tf.app.flags.DEFINE_integer('batch_size', CRNN_BATCH_SIZE, '')
    tf.app.flags.DEFINE_string('charset', CRNN_CHARSET_FILE, '')    # 字符集
