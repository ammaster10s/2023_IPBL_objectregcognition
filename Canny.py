import cv2
import numpy as np

def testCanny(img, params):
    med_val = np.median(img)
    sigma = 0.33
    min_val = int(max(0, (1.0 - sigma) * med_val))
    max_val = int(max(255, (1.0 + sigma) * med_val))

    img_edge = cv2.Canny(img, threshold1 = min_val, threshold2 = max_val)

    cv2.imshow('result', img_edge)
    cv2.waitKey(0)

    # cv2.imwrite('image/automatic/image-canny-automatic.jpg', img_edge)

def Canny(img, params):

    med_val = np.median(img)
    sigma = 0.33
    min_val = int(max(0, (1.0 - sigma) * med_val))
    max_val = int(max(255, (1.0 + sigma) * med_val))

    img_edge = cv2.Canny(img, threshold1 = min_val, threshold2 = max_val)

    return img_edge

if __name__ == '__main__':
    dummy = 0
    filePath = './train_img/0005.jpg'
    img = cv2.imread(filePath)
    testCanny(img ,dummy)

