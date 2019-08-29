# coding:utf-8
import logging

import numpy as np

from module.ctpn.utils.bbox.nms import nms  # 原来是 from nms import nms
from .text_connect_cfg import Config as TextLineCfg
from .text_proposal_connector import TextProposalConnector
from .text_proposal_connector_oriented import TextProposalConnectorOriented

logger = logging.getLogger("text detector")


class TextDetector:
    def __init__(self, DETECT_MODE="H"):
        self.mode = DETECT_MODE
        if self.mode == "H":
            self.text_proposal_connector = TextProposalConnector()
        elif self.mode == "O":
            self.text_proposal_connector = TextProposalConnectorOriented()

    def detect(self, text_proposals, scores, size):

        logger.debug("detect - text proposal shape:%r", text_proposals.shape)
        logger.debug("detect - score:%r", scores.shape)
        logger.debug("detect - size:%r", size)

        # 删除得分较低的proposal,TEXT_PROPOSALS_MIN_SCORE=0.7,这个是前景的置信度
        keep_inds = np.where(scores > TextLineCfg.TEXT_PROPOSALS_MIN_SCORE)[0]
        logger.debug("只保留scores成绩>0.7的anchor:保留了%d个", len(keep_inds))
        text_proposals, scores = text_proposals[keep_inds], scores[keep_inds]

        # 按得分排序，按照得分对text_proposals进行了排序，argsoft是从小到大，[::-1]是再反过来，即从大到小
        sorted_indices = np.argsort(scores.ravel())[::-1]
        text_proposals, scores = text_proposals[sorted_indices], scores[sorted_indices]
        logger.debug("小框一共%d个", len(text_proposals))

        # 对proposal做nms
        # nms就是刨除掉那些比最优次优的proposals，
        # 当然如果差很多就不刨除，反倒保留着，根据阈值
        keep_inds = nms(np.hstack((text_proposals, scores)), TextLineCfg.TEXT_PROPOSALS_NMS_THRESH)
        text_proposals, scores = text_proposals[keep_inds], scores[keep_inds]
        logger.info("NMS过滤后的小框有%d个", len(text_proposals))

        # 获取检测结果，返回的是4个点的4边形框
        text_recs = self.text_proposal_connector.get_text_lines(text_proposals, scores, size)
        logger.debug("探测出来的备选框：%d", len(text_recs))

        keep_inds = self.filter_boxes(text_recs)
        logger.info("需要保留(宽/高比>0.5,置信度>0.9,宽>32)下来的框%d个", len(keep_inds))
        return text_recs[keep_inds]

    #     0  1  2  3  4  5  6  7
    # box[x1,y1,x2,y2,x3,y3,x4,y4]
    def filter_boxes(self, boxes):
        heights = np.zeros((len(boxes), 1), np.float)
        widths = np.zeros((len(boxes), 1), np.float)
        scores = np.zeros((len(boxes), 1), np.float)
        index = 0
        for box in boxes:
            # |y3 - y1| + |y4 - y2|
            # 高度取个平均数
            h = (abs(box[5] - box[1]) + abs(box[7] - box[3])) / 2.0 + 1
            heights[index] = h
            # 宽度取个平均数
            w = (abs(box[2] - box[0]) + abs(box[6] - box[4])) / 2.0 + 1
            widths[index] = w
            scores[index] = box[8]
            index += 1
            # logger.debug("宽[%d]高[%d]置信度[%f]",w,h,box[8])

        # MIN_RATIO=0.5 ，
        # LINE_MIN_SCORE = 0.9 ，
        # TEXT_PROPOSALS_WIDTH = 16
        # MIN_NUM_PROPOSALS = 2
        # 宽/高比>0.5 and 置信度>0.9 and 宽>32
        return np.where(
            (widths / heights > TextLineCfg.MIN_RATIO) &  # 0.5
            (scores > TextLineCfg.LINE_MIN_SCORE) &  # 0.9
            (widths > (TextLineCfg.TEXT_PROPOSALS_WIDTH * TextLineCfg.MIN_NUM_PROPOSALS)))[0]  # 16 * 2
