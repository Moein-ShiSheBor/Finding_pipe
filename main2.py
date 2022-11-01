import cv2
import os
import numpy as np
from matplotlib import pyplot as plt
import diplib as dip

IMG_PATH = os.path.join('IMG', 'full.jpg')
img = cv2.imread(IMG_PATH)

resized_img = cv2.resize(img, (int(img.shape[1]), int(img.shape[0])))
# resized_img = cv2.resize(img, (int(img.shape[1]), int(img.shape[0])))

gray = cv2.cvtColor(resized_img, cv2.COLOR_BGR2GRAY)

# blur = cv2.GaussianBlur(gray, (3, 3), 0)
blur = cv2.GaussianBlur(gray, (5, 5), 0)

# canny = cv2.Canny(blur, threshold1=180, threshold2=210)
canny = cv2.Canny(blur, threshold1=150, threshold2=210)

culled_edge = canny.copy()

cv2.imshow('Frame View', culled_edge)
cv2.waitKey(0)
cv2.destroyAllWindows()

# cv2.imwrite('sample1.jpg', canny)
