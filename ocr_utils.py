import cv2
import logging
import base64

logger = logging.getLogger("OCR Utils")

# 输入是[[x1,y1,x2,y2,x3,y3,x4,y4],]
def crop_small_images(img,polygens):
    logger.debug("图像：%r" , img.shape)
    cropped_images = []
    for pts in polygens:
        # crop_img = img[y:y+h, x:x+w]
        logger.debug("子图坐标：%r",pts)
        crop_img = img[pts[3]:pts[5], pts[0]:pts[2]]
        cropped_images.append(crop_img)
    return cropped_images


def tobase64(data):

    if type(data)==list:
        result = []
        for d in data:
            _,buf = cv2.imencode('.jpg', d)
            result.append(str(base64.b64encode(buf),'utf-8'))
        return result

    _,d = cv2.imencode('.jpg', data)
    return str(base64.b64encode(d), 'utf-8')

if __name__ == '__main__':

    image = cv2.imread("test/test.png")
    polygens = [
        [0,0,100,0,100,100,0,100]
    ]

    for img in crop_small_images(image,polygens):
        print("子图：",img.shape)
        # cv2.imshow("cropped", img)
        # cv2.waitKey(0)
        tobase64(img)
