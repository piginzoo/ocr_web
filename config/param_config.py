import tensorflow as tf

################################################################
#
#   定义CTPN、CRNN的共同参数
#
################################################################

# 定义相关的目录，这个是为了方便3个项目集成，所以需要绝对路径，方便import，不是为了找文件，找文件用相对路径就好，是为了python import
CTPN_HOME = ['ctpn']

# CTPN常用参数
CTPN_MODEL_FILE = "ctpn-2019-05-28-22-32-39-2901.ckpt"  # 定义模型目录，如果FILE被定义，直接加载FILE，否则，挑选最新的模型加载(即在checkpoint文件中记录的）

CTPN_MODEL_DIR = "../models"
CTPN_DRAW = True
CTPN_SAVE = False  # 是否保存识别的坐标
CTPN_EVALUATE = False  # 是否提供评估
CTPN_SPLIT = True  # 是否保留小框(CTPN网络识别结果）
CTPN_PRED_DIR = "data/pred"  # 保存的内容存放的目录
CTPN_TEST_DIR = "data/test"  #

# CRNN常用参数
CRNN_HOME = ['crnn']
CRNN_MODEL_FILE = "crnn_2019-06-27-05-06-48.ckpt-58000"
# CRNN_MODEL_FILE= "LATEST"
CRNN_MODEL_DIR = "../models"
CRNN_CHARSET_FILE = "../crnn/charset.3770.txt"
CRNN_BATCH_SIZE = 128

# 通用的调试开关
DEBUG = True


# 把没必要的参数都关闭
def disable_debug_flags():
    tf.app.flags.FLAGS.remove_flag_values({'evaluate': True})
    tf.app.flags.FLAGS.remove_flag_values({'split': True})
    tf.app.flags.FLAGS.remove_flag_values({'draw': True})
    tf.app.flags.FLAGS.remove_flag_values({'save': True})
    tf.app.flags.DEFINE_boolean('evaluate', False, '')  # 是否进行评价（你可以光预测，也可以一边预测一边评价）
    tf.app.flags.DEFINE_boolean('split', True, '')  # 是否对小框做出评价，和画到图像上
    tf.app.flags.DEFINE_boolean('draw', True, '')  # 是否把gt和预测画到图片上保存下来，保存目录也是pred_home
    tf.app.flags.DEFINE_boolean('save', False, '')  # 是否保存输出结果（大框、小框信息都要保存），保存到pred_home目录里面去
