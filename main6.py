import cv2
import os
import numpy as np
from matplotlib import pyplot as plt
import diplib as dip

IMG_PATH_REAL = os.path.join('IMG', 'full-e.jpg')
# IMG_PATH_REAL = os.path.join('IMG', '13.png')
img_real = cv2.imread(IMG_PATH_REAL)

resized_img_real = cv2.resize(img_real, (int(img_real.shape[1]), int(img_real.shape[0])))

gray_real = cv2.cvtColor(resized_img_real, cv2.COLOR_BGR2GRAY)

blur_real = cv2.GaussianBlur(gray_real, (5, 5), 0)

canny_real = cv2.Canny(blur_real, threshold1=180, threshold2=230)
# canny_real = cv2.Canny(blur_real, threshold1=10, threshold2=50)

rho = 1  # distance resolution in pixels of the Hough grid
theta = np.pi / 180  # angular resolution in radians of the Hough grid
threshold = 15  # minimum number of votes (intersections in Hough grid cell)
min_line_length = 65  # minimum number of pixels making up a line
max_line_gap = 20  # maximum gap in pixels between connectable line segments
line_image = np.copy(resized_img_real) * 0  # creating a blank to draw lines on

lines = cv2.HoughLinesP(canny_real, rho, theta, threshold, np.array([]),
                        min_line_length, max_line_gap)


def intended_angle(x1, y1, x2, y2):
    delta_y = y2 - y1
    delta_x = x2 - x1

    angle_degrees = np.arctan(delta_y / delta_x) * 180 / np.pi

    angle_degrees = abs(angle_degrees)

    print('--------------')
    print(angle_degrees)

    if (angle_degrees < 90 and angle_degrees > 45) or (angle_degrees > 90 and angle_degrees < 135):
        return True
    return False


for line in lines:
    for x1, y1, x2, y2 in line:
        if intended_angle(x1, y1, x2, y2):
            print(line)
            cv2.line(line_image, (x1, y1), (x2, y2), (255, 0, 0), 5)

lines_edges = cv2.addWeighted(img_real, 0.8, line_image, 1, 0)

print('************')
# print(canny_real.size)
print(img_real.shape)

cv2.imshow("Result", lines_edges)
cv2.imshow("Result line", line_image)
cv2.imshow("Edge real", canny_real)

# path = "C:/Users/MooNice/Desktop/Edge_Detection/1/sample_pics"
#
# cv2.imwrite(os.path.join(path, 'Result.jpg'), lines_edges)
# cv2.imwrite(os.path.join(path, 'Edge.jpg'), canny_real)

cv2.waitKey(0)
cv2.destroyAllWindows()
