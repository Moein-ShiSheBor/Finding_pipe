import cv2
import os
import numpy as np
from matplotlib import pyplot as plt
import diplib as dip

IMG_PATH_REAL = os.path.join('IMG', 'full-e.jpg')
img_real = cv2.imread(IMG_PATH_REAL)

IMG_PATH_SAMPLE = os.path.join('IMG', '13.png')
img_sample = cv2.imread(IMG_PATH_SAMPLE)

resized_img_real = cv2.resize(img_real, (int(img_real.shape[1]), int(img_real.shape[0])))
resized_img_sample = cv2.resize(img_sample, (int(img_sample.shape[1] / 2), int(img_sample.shape[0] / 2)))

gray_real = cv2.cvtColor(resized_img_real, cv2.COLOR_BGR2GRAY)
gray_sample = cv2.cvtColor(resized_img_sample, cv2.COLOR_BGR2GRAY)

blur_real = cv2.GaussianBlur(gray_real, (5, 5), 0)
blur_sample = cv2.GaussianBlur(gray_sample, (5, 5), 0)

canny_real = cv2.Canny(blur_real, threshold1=200, threshold2=250)
canny_sample = cv2.Canny(blur_sample, threshold1=20, threshold2=45)

orb_real = cv2.ORB_create(nfeatures=1500)
orb_sample = cv2.ORB_create(nfeatures=1500)

keypoints_real, descriptors_real = orb_real.detectAndCompute(canny_real, None)
keypoints_sample, descriptors_sample = orb_real.detectAndCompute(canny_sample, None)

canny_real = cv2.drawKeypoints(canny_real, keypoints_real, None)
canny_sample = cv2.drawKeypoints(canny_sample, keypoints_sample, None)

##

bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

matches = bf.match(descriptors_sample, descriptors_real)
matches = sorted(matches, key=lambda x: x.distance)

print(len(matches))

for m in matches[:20]:
    print(m.distance)

matching_result = cv2.drawMatches(canny_sample, keypoints_sample, canny_real, keypoints_real, matches[:20], None)

# min_threshold = 100
# canny_real_dib = dip.BinaryAreaOpening(canny_real > 0, min_threshold)
# canny_sample_dib = dip.BinaryAreaOpening(canny_sample > 0, min_threshold)
#
#
# cv2.imshow("Image real dib", canny_real_dib)
# cv2.imshow("Image sample dib", canny_sample_dib)

cv2.imshow("Image real", canny_real)
cv2.imshow("Image sample", canny_sample)
cv2.imshow("Result", matching_result)
cv2.waitKey(0)
cv2.destroyAllWindows()
