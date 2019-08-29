import logging

import cv2
import tensorflow as tf

logger = logging.getLogger("PreProcessUtil")


def image_resize_with_pad(image, target_height, target_width, pad_val):
    # logger.debug("开始做padding转换...")

    # logger.debug("image.shape:%r",image.shape)
    # logger.debug("target_height:%r",target_height)
    # logger.debug("target_width:%r",target_width)
    # logger.debug("pad_val:%r",pad_val)

    height, width, _ = image.shape

    x_scale = target_width / width
    y_scale = target_height / height
    # logger.debug("x_scale:%r", x_scale)
    # logger.debug("y_scale:%r", y_scale)

    min_scale = min(x_scale, y_scale)
    # logger.debug("scale:%r", min_scale)

    image = cv2.resize(image, None, fx=min_scale, fy=min_scale, interpolation=cv2.INTER_AREA)
    # logger.debug("resize_image.shape:%r", image.shape)

    after_resize_height, after_resize_width, _ = image.shape

    # top,bottom,left,right对应边界的像素数目
    top = int(round((target_height - after_resize_height) / 2))
    bottom = target_height - top - after_resize_height

    left = int(round((target_width - after_resize_width) / 2))
    right = target_width - left - after_resize_width

    image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[255, 255, 255])
    # logger.debug("padding后的图像:%r",image.shape)

    return image


def test_resize_image():
    decode_jpeg_data = tf.placeholder(dtype=tf.string)
    image_raw_data = tf.gfile.FastGFile('./input.jpg', 'rb').read()

    # 将图像以jpeg的格式解码从而得到图像对应的三维矩阵
    # tf.image_decode_png 函数对png格式图形进行解码。解码之后得到一个张量
    # tf.image_decode_jpeg 函数对jpeg格式图形进行解码。
    img_data = tf.image.decode_jpeg(decode_jpeg_data, channels=3)

    # padding
    img_data2 = tf.py_func(image_resize_with_pad, [img_data, 32, 512, 255], [tf.uint8])
    img_data3 = tf.convert_to_tensor(tf.cast(img_data2, tf.int32), name='img_data2')

    with tf.Session() as sess:
        img = sess.run(img_data3, feed_dict={decode_jpeg_data: image_raw_data})
        print(img.shape)


if __name__ == "__main__":
    test_resize_image()
