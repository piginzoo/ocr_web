# import tensorflow as tf
#
# from config import param_config
#
#
# def init_params():
#     tf.app.flags.DEFINE_string('image_name', '', '')  # 被预测的图片名字，为空就预测目录下所有的文件
#
#     tf.app.flags.DEFINE_string('test_images_dir', '', '')
#     tf.app.flags.DEFINE_string('test_labels_dir', '', '')
#     tf.app.flags.DEFINE_string('test_labels_split_dir', '', '')
#
#     tf.app.flags.DEFINE_string('IP', '127.0.0.1', '')
#     tf.app.flags.DEFINE_string('imgn', 'test.JPG', '')
#
#     # 共享的
#     tf.app.flags.DEFINE_boolean('debug', param_config.DEBUG, '')  # 是否调试
#
#     # 这个是为了兼容
#     # gunicorn -w 2 -k gevent web.api_server:app -b 0.0.0.0:8080
#     tf.app.flags.DEFINE_string('worker-class', 'gevent', '')
#     tf.app.flags.DEFINE_integer('workers', 1, '')
#     tf.app.flags.DEFINE_string('bind', '0.0.0.0:8080', '')
#     tf.app.flags.DEFINE_integer('timeout', 60, '')
#     tf.app.flags.DEFINE_string('preload', '', '')
#     tf.app.flags.DEFINE_integer('worker-connections', 1000, '')
#
#     # ctpn的
#     tf.app.flags.DEFINE_string('ctpn_model_dir', param_config.CTPN_MODEL_DIR, '')  # model的存放目录，会自动加载最新的那个模型
#     tf.app.flags.DEFINE_string('ctpn_model_file', param_config.CTPN_MODEL_FILE, '')  # 为了支持单独文件，如果为空，就挑选最新的model
#     tf.app.flags.DEFINE_boolean('evaluate', param_config.CTPN_EVALUATE, '')  # 是否进行评价（你可以光预测，也可以一边预测一边评价）
#     tf.app.flags.DEFINE_boolean('split', param_config.CTPN_SPLIT, '')  # 是否对小框做出评价，和画到图像上
#     tf.app.flags.DEFINE_boolean('draw', param_config.CTPN_DRAW, '')  # 是否把gt和预测画到图片上保存下来，保存目录也是pred_home
#     tf.app.flags.DEFINE_boolean('save', param_config.CTPN_SAVE, '')  # 是否保存输出结果（大框、小框信息都要保存），保存到pred_home目录里面去
#     tf.app.flags.DEFINE_string('pred_dir', param_config.CTPN_PRED_DIR, '')  # 预测后的结果的输出目录
#     tf.app.flags.DEFINE_string('test_dir', param_config.CTPN_TEST_DIR, '')  # 预测后的结果的输出目录
#
#     # crnn的
#     tf.app.flags.DEFINE_string('crnn_model_dir', param_config.CRNN_MODEL_DIR, '')
#     tf.app.flags.DEFINE_string('crnn_model_file', param_config.CRNN_MODEL_FILE, '')
#     tf.app.flags.DEFINE_integer('batch_size', param_config.CRNN_BATCH_SIZE, '')
#     tf.app.flags.DEFINE_string('charset', param_config.CRNN_CHARSET_FILE, '')  # 字符集
#     tf.app.flags.DEFINE_string('resize_mode', 'PAD', '')
#
#
# init_params()