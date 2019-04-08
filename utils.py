import cv2

def crop_small_images(img,polygens):
    print("图像：" , img.shape)
    cropped_images = []
    for pts in polygens:
        # crop_img = img[y:y+h, x:x+w]
        print("子图坐标：",pts)
        crop_img = img[pts[1]:pts[3], pts[0]:pts[2]]
        cropped_images.append(crop_img)
    return cropped_images

if __name__ == '__main__':

    img = cv2.imread("test/test.png")
    polygens = [
        [0,0,100,100],
        [100,100,200,200]
    ]

    for img in crop_small_images(img,polygens):
        print("子图：",img.shape)
        cv2.imshow("cropped", img)
        cv2.waitKey(0)