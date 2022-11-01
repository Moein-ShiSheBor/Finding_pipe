import cv2
import os
import numpy as np


def intended_angle(x1, y1, x2, y2):
    delta_y = y2 - y1
    delta_x = x2 - x1

    angle_degrees = np.arctan(delta_y / delta_x) * 180 / np.pi

    angle_degrees = abs(angle_degrees)

    print('--------------')
    print(angle_degrees)

    if (angle_degrees < 120 and angle_degrees > 30) or (angle_degrees > 35 and angle_degrees < 160):
        return True
    return False


for i in range(2186):
    image_name = f'frame{i+1}.jpg'
    IMG_PATH_REAL = os.path.join('frame_save_crop', image_name)
    img_real = cv2.imread(IMG_PATH_REAL)

    resized_img_real = cv2.resize(img_real, (int(img_real.shape[1]), int(img_real.shape[0])))

    gray_real = cv2.cvtColor(resized_img_real, cv2.COLOR_BGR2GRAY)

    blur_real = cv2.GaussianBlur(gray_real, (5, 5), 0)

    canny_real = cv2.Canny(blur_real, threshold1=130, threshold2=150)

    rho = 1  # distance resolution in pixels of the Hough grid
    theta = np.pi / 180  # angular resolution in radians of the Hough grid
    threshold = 15  # minimum number of votes (intersections in Hough grid cell)
    min_line_length = 65  # minimum number of pixels making up a line
    max_line_gap = 20  # maximum gap in pixels between connectable line segments
    line_image = np.copy(resized_img_real) * 0  # creating a blank to draw lines on

    lines = cv2.HoughLinesP(canny_real, rho, theta, threshold, np.array([]),
                            min_line_length, max_line_gap)

    for line in lines:
        for x1, y1, x2, y2 in line:
            if intended_angle(x1, y1, x2, y2):
                # print(line)
                cv2.line(line_image, (x1, y1), (x2, y2), (255, 0, 0), 5)

    lines_edges = cv2.addWeighted(img_real, 0.8, line_image, 1, 0)

    cv2.imshow(f"Result {i + 1}", lines_edges)
    # cv2.imshow("Result line", line_image)
    cv2.imshow(f"Edge {i + 1}", canny_real)

    path = "C:/Users/MooNice/Desktop/Edge_Detection/1/frame_result_new_angle"

    cv2.imwrite(os.path.join(path, f'Result {i + 1}.jpg'), lines_edges)
    cv2.imwrite(os.path.join(path, f'Edge {i + 1}.jpg'), canny_real)

cv2.waitKey(0)
cv2.destroyAllWindows()
