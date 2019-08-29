import logging
import os

import cv2
import numpy as np

from module.crnn.config import config

logger = logging.getLogger("ImageUtil")


# 读取图像
def read_image_file(image_file: str, image_type: int = 1):
    if image_type == 1:
        # 彩色图像
        img = cv2.imread(image_file, cv2.IMREAD_COLOR)
    elif image_type == 0:
        # 灰度图像
        img = cv2.imread(image_file, cv2.IMREAD_GRAYSCALE)
    else:
        raise ValueError('the image_type value be 0(gray) or 1(color). ')

    return img


# 图像缩放，高度都是32
def resize_batch_image(image_list: list, output_size: tuple, resize_mode):
    out_height, out_width = output_size
    target_image_list = []
    # 因为模型要求，output_size 必须为32
    if out_height != config.INPUT_SIZE[0]:
        raise ValueError('the height of output_size must be 32')

    # 因为模型要求，output_widt 必须被4整除
    if out_width % config.WIDTH_REDUCE_TIMES != 0:
        raise ValueError('the width of output_size % 4 must be 0')

    # 强制缩放
    for img in image_list:
        out_image = None
        # logger.debug("resize图片,原始尺寸:%r，Resize尺寸：%r",img.shape,(out_width, out_height))
        if resize_mode == config.RESIZE_MODE_FIX:
            out_img = cv2.resize(img, (out_width, out_height), interpolation=cv2.INTER_AREA)
        elif resize_mode == config.RESIZE_MODE_PAD:
            out_img = resize_by_height_with_padding(img, out_height, out_width)
        else:
            raise ValueError("识别不出来的Resize模式：%s", resize_mode)

        # 调试用，可删
        # import os,random
        # if not os.path.exists("./data/temp/"): os.makedirs("./data/temp/")
        # cv2.imwrite("data/temp/"+str(random.randint(1,1000000))+".png",out_img)

        target_image_list.append(out_img)

    return target_image_list


# 获取图像列表最大宽度
def get_max_width(image_list):
    max_width = 0
    for image in image_list:
        width = image.shape[1]
        if max_width < width:
            max_width = width

    return max_width


# 获取图像列表宽度的中位数
def get_median_width(image_list):
    max_width = 0
    width_list = [image.shape[1] for image in image_list]
    return int(np.median(np.array(width_list)))


def resize_by_height_with_padding(image, target_height, target_width):
    h, w, _ = image.shape
    scale = target_height / h
    target_image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

    h, w, _ = target_image.shape
    # 缩放完，超宽了，截掉
    if w > target_width:
        return target_image[:, :target_width, :]

    # print((target_width, target_width - w))
    target_image = np.pad(target_image,
                          pad_width=((0, 0),  # 高度不动
                                     (0, target_width - w),  # 宽度(before, after),左添加0，右边添加512-目前宽度
                                     (0, 0)),  # Channel不动
                          mode="constant",
                          constant_values=(255))

    # logger.debug("原图:%r，Resize图：%r",image.shape,target_image.shape)
    return target_image


# 根据最大size获取合适的缩放后的图像,废弃了，处理的比较复杂，简化之,6.7,piginzoo
def get_scaled_image_no_padding(image, max_size):
    max_height, max_width = max_size
    src_height, src_width, _ = image.shape

    # 获取图像缩放比例
    x_scale = max_width / src_width
    y_scale = max_height / src_height
    target_scala = min(x_scale, y_scale)

    # 获取目标比例
    scaled_width = int(round(src_width * target_scala))

    # 高度处理
    # scaled_height = int(round(src_height * target_scala))
    # if scaled_height != 32:
    #     target_height = 32
    target_height = config.INPUT_SIZE[0]

    # 宽度处理
    target_width = get_valid_width(scaled_width, max_width)

    # 图像缩放
    target_img = cv2.resize(image, (target_width, target_height), interpolation=cv2.INTER_AREA)
    return target_img


# 获取有效的宽度值
def get_valid_width(width, max_width):
    if max_width % config.WIDTH_REDUCE_TIMES != 0:
        raise ValueError('max_width % 4 should be 0')

    if width > max_width:
        return max_width

    target_width = (width // config.WIDTH_REDUCE_TIMES) * config.WIDTH_REDUCE_TIMES
    if (width > target_width) and (target_width + config.WIDTH_REDUCE_TIMES <= max_width):
        target_width += config.WIDTH_REDUCE_TIMES
    return target_width


if __name__ == '__main__':
    img_dir = r'D:\workspace\gitCDC\ocr\hang\CRNN_Tensorflow\data\train'

    # 读取图像
    image_list = []
    for i in range(10):
        image_path = os.path.join(img_dir, str(i) + ".png")
        image = read_image_file(image_path)
        image_list.append(image)

    before_resize = [img.shape for img in image_list]

    # 缩放图像
    out_resize = (32, 1024)
    print("图像out_size = ", out_resize)

    # RESIZE_FORCE、RESIZE_MAX、RESIZE_FREE
    resize_mode = "RESIZE_MAX"
    print("图像resize_mode = " + resize_mode)
    resize_list = resize_batch_image(image_list, resize_mode, out_resize)

    after_resize = [img.shape for img in resize_list]

    print("图像resize结果如下:")
    cnt = len(before_resize)
    for idx in range(cnt):
        one = before_resize[idx]
        two = after_resize[idx]
        print("%r ==> %r" % (one, two))
