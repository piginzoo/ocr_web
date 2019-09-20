# -*- coding: utf-8 -*-
"""
    说明：
"""

import os

import cv2
import tensorflow as tf

from utils.ctpn.utils import stat
from utils.ctpn.utils.dataset import data_provider as data_provider
from utils.ctpn.utils.evaluate.evaluator import *
from utils.ctpn.utils.rpn_msr.config import Config

logger = logging.getLogger("ctpn_handle")

FLAGS = tf.app.flags.FLAGS


def cls_prob_val_reshape_debug_stat(cls_prob_val):
    cls_prob_for_debug = cls_prob_val.reshape(1,
                                              cls_prob_val.shape[1],  # H
                                              cls_prob_val.shape[2] // Config.NETWORK_ANCHOR_NUM,  # W
                                              Config.NETWORK_ANCHOR_NUM,  # 每个点上扩展的10个anchors
                                              -1)  # <---2个值, 0:背景概率 1:前景概率
    _stat = stat(cls_prob_for_debug[:, :, :, :, 1].reshape(-1, 1))  # 去掉背景，只保留前景，然后传入统计
    return _stat


RED = (255, 0, 0)
GREEN = (0, 255, 0)
GRAY = (50, 50, 50)
BLUE = (0, 0, 255)

# 测试目录下，包含了3个子路径：放图片的images,放标签的labels，放小框标签的split
IMAGE_PATH = "images"  # 要文本检测的图片
LABEL_PATH = "labels"  # 大框数据，
SPLIT_PATH = "split"  # 小框数据
# 输出的路径
PRED_DRAW_PATH = "draws"  # 画出来的数据
PRED_BBOX_PATH = "detect.bbox"  # 探测的小框
PRED_GT_PATH = "detect.gt"  # 探测的大框


# 把框画到图片上
# 注意：image是RGB格式的
def draw(image, boxes, color, thick=1):
    if len(boxes) == 0: return

    # 先将RGB格式转成BGR，也就是OpenCV要求的格式
    # image = image[:,:,::-1]
    if boxes.shape[1] == 4:  # 矩形
        for box in boxes:
            box = box.astype(np.int32)
            cv2.rectangle(image,
                          (box[0], box[1]),
                          (box[2], box[3]),
                          color=color,
                          thickness=thick)
        return
    if boxes.shape[1] == 8:  # 四边形
        for box in boxes:
            cv2.polylines(image,
                          [box[:8].astype(np.int32).reshape((-1, 2))],
                          True,
                          color=color,
                          thickness=thick)
        return

    logger.error("画图失败，无效的Shape:%r", boxes.shape)


# 根据图片文件名，得到，对应的标签文件名，可能是split的小框的(矩形4个值)，也可能是4个点的大框的（四边形8个值）
def get_gt_label_by_image_name(image_name, label_path):
    label_name = os.path.splitext(os.path.basename(image_name))  # ['123','png'] 123.png

    if len(label_name) != 2:
        logger.error("图像文件解析失败：image_name[%s],label_name[%s]", image_name, label_name)
        return None

    label_name = label_name[0]  # /usr/test/123.png => 123
    label_name = os.path.join(label_path, label_name + ".txt")
    if not os.path.exists(label_name):
        logger.error("标签文件不存在：%s", label_name)
        return None

    bbox = data_provider.load_big_GT(label_name)
    logger.debug("加载了%d个GT(4个点,8个值)", len(bbox))

    return np.array(bbox)


# 保存预测的输出结果，保存大框和小框，都用这个函数，保存大框的时候不需要scores这个参数
def save(path, file_name, data, scores=None):
    # 输出
    logger.debug("保存坐标文件，目录：%s，名字：%s", path, file_name)
    with open(os.path.join(path, file_name), "w") as f:
        for i, one in enumerate(data):
            line = ",".join([str(value) for value in one])
            if scores is not None:
                line += "," + str(scores[i])
            line += "\r\n"
            f.writelines(line)
    logger.info("预测结果保存完毕：%s/%s", path, file_name)


def post_detect(bbox_small, boxes_big, image_name, original_img, scores):
    draw_image = None
    f1_value = None

    # 输出的路径
    pred_draw_path = os.path.join(FLAGS.pred_dir, PRED_DRAW_PATH)
    pred_gt_path = os.path.join(FLAGS.pred_dir, PRED_GT_PATH)
    pred_bbox_path = os.path.join(FLAGS.pred_dir, PRED_BBOX_PATH)
    label_path = os.path.join(FLAGS.test_dir, LABEL_PATH)
    split_path = os.path.join(FLAGS.test_dir, SPLIT_PATH)
    if not os.path.exists(pred_bbox_path): os.makedirs(pred_bbox_path)
    if not os.path.exists(pred_draw_path): os.makedirs(pred_draw_path)
    if not os.path.exists(pred_gt_path): os.makedirs(pred_gt_path)
    # 如果关注小框就把小框画上去
    if FLAGS.draw:
        if FLAGS.split:
            draw_image = original_img.copy()
            # 把预测小框画上去
            draw(draw_image, bbox_small, GREEN)
            logger.debug("将预测出来的小框画上去了")

            split_box_labels = get_gt_label_by_image_name(image_name, split_path)
            if split_box_labels:
                draw(draw_image, split_box_labels, BLUE)
                logger.debug("将样本的小框画上去了")

        # 来！把预测的大框画到图上，输出到draw目录下去，便于可视化观察
        draw(draw_image, boxes_big, color=RED, thick=1)
        logger.debug("将大框画上去了")

        out_image_path = os.path.join(pred_draw_path, os.path.basename(image_name))
        cv2.imwrite(out_image_path, draw_image)
        logger.debug("绘制预测和GT到图像完毕：%s", out_image_path)
    # 是否保存预测结果（包括大框和小框）=> data/pred目录
    if FLAGS.save:
        file_name = os.path.splitext(os.path.basename(image_name))[0] + ".txt"
        # 输出大框到文件
        save(
            pred_gt_path,
            file_name,
            boxes_big
        )
        logger.debug("保存了大框的坐标到：%s/%s", pred_gt_path, file_name)

        # 输出小框到文件

        save(
            pred_bbox_path,
            file_name,
            bbox_small,
            scores
        )
        logger.debug("保存了小框的坐标到：%s/%s", pred_bbox_path, file_name)
    # 是否做评价
    if FLAGS.evaluate:
        # 对8个值（4个点）的任意四边形大框做评价
        big_box_labels = get_gt_label_by_image_name(image_name, label_path)
        if big_box_labels is not None:
            logger.debug("找到图像（%s）对应的大框样本（%d）个，开始评测", image_name, len(big_box_labels))
            metrics = evaluate(big_box_labels, boxes_big[:, :8], conf())
            # _image['F1'] = metrics['hmean']
            f1_value = metrics['hmean']
            logger.debug("大框的评价：%r", metrics)
            draw(original_img, big_box_labels[:, :8], color=GRAY, thick=2)

        # 对4个值（2个点）的矩形小框做评价
        if FLAGS.split:
            split_box_labels = get_gt_label_by_image_name(image_name, split_path)
            if split_box_labels is not None:
                logger.debug("找到图像（%s）对应的小框split样本（%d）个，开始评测", image_name, len(split_box_labels))
                metrics = evaluate(split_box_labels, bbox_small, conf())
                logger.debug("小框的评价：%r", metrics)
                logger.debug("将小框标签画到图片上去")
                draw(original_img, split_box_labels[:, :4], color=GRAY, thick=1)

    return draw_image, f1_value
