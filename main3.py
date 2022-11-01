import cv2
import os
import numpy as np
from matplotlib import pyplot as plt
import diplib as dip

IMG_PATH = os.path.join('IMG', 'full-e.jpg')
img = cv2.imread(IMG_PATH)

resized_img = cv2.resize(img, (int(img.shape[1]), int(img.shape[0])))

gray = cv2.cvtColor(resized_img, cv2.COLOR_BGR2GRAY)

blur = cv2.GaussianBlur(gray, (5, 5), 0)

canny = cv2.Canny(blur, threshold1=200, threshold2=250)

orb = cv2.ORB_create(nfeatures=1500)

keypoints, descriptors = orb.detectAndCompute(canny, None)

canny = cv2.drawKeypoints(canny, keypoints, None)

cv2.imshow("Image", canny)
cv2.waitKey(0)
cv2.destroyAllWindows()
