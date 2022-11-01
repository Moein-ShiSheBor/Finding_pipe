import math

import cv2
import os
import numpy as np


def intended_angle_ver(x1, y1, x2, y2):
    delta_y = y2 - y1
    delta_x = x2 - x1

    angle_degrees_ver = np.arctan(delta_y / delta_x) * 180 / np.pi

    angle_degrees_ver = abs(angle_degrees_ver)

    # print('--------------')
    # print(angle_degrees_hor)

    if (angle_degrees_ver < 160 and angle_degrees_ver > 20):
        return True
    return False


def intended_angle_hor(x1, y1, x2, y2):
    delta_y = y2 - y1
    delta_x = x2 - x1

    angle_degrees_hor = np.arctan(delta_y / delta_x) * 180 / np.pi

    angle_degrees_hor = abs(angle_degrees_hor)

    # print('--------------')
    # print(angle_degrees_ver)

    if (angle_degrees_hor < 30):
        return True
    return False


def line_coefficient(x1, y1, x2, y2):
    a = (y2 - y1) / (x2 - x1)
    b = y1 - (a * x1)

    return a, b


def collision_point(a1, b1, a2, b2):
    x = (b2 - b1) / (a1 - a2)
    y = a1 * x + b1

    return x, y


def out_of_range(x, y):
    return 0 < x < int(resized_img_real_ver.shape[0]) and 0 < y < int(resized_img_real_ver.shape[1])


def len_of_line(y2, y1, x2, x1):
    return math.sqrt((y2 - y1) ** 2 + (x2 - x1) ** 2)


def check_hor_length(i, j, k):
    x1 = i[0][0]
    y1 = i[0][1]
    x2 = i[0][2]
    y2 = i[0][3]

    x11 = j[0][0]
    y11 = j[0][1]
    x22 = j[0][2]
    y22 = j[0][3]

    x111 = k[0][0]
    y111 = k[0][1]
    x222 = k[0][2]
    y222 = k[0][3]

    a1, b1 = line_coefficient(x1, y1, x2, y2)
    a2, b2 = line_coefficient(x11, y11, x22, y22)
    a3, b3 = line_coefficient(x111, y111, x222, y222)

    col_x_1_3, col_y_1_3 = collision_point(a1, b1, a3, b3)
    col_x_2_3, col_y_2_3 = collision_point(a2, b2, a3, b3)

    if out_of_range(col_x_1_3, col_y_1_3) or out_of_range(col_x_2_3, col_y_2_3):
        # print(col_x_1_3, col_y_1_3)
        # print(col_x_2_3, col_y_2_3)

        hor_len = len_of_line(col_y_2_3, col_y_1_3, col_x_2_3, col_x_1_3)

        # print(hor_len)
        # print('---------------')
        return 450 < hor_len < 600


for i in range(1):
    global resized_img_real_ver
    image_name = f'frame38-1.jpg'
    # image_name = f'frame35.jpg'
    IMG_PATH_REAL = os.path.join('test_frame', image_name)
    img_real_ver = cv2.imread(IMG_PATH_REAL)
    img_real_hor = cv2.imread(IMG_PATH_REAL)

    # print(int(img_real_ver.shape[1]), int(img_real_ver.shape[0]))

    resized_img_real_ver = cv2.resize(img_real_ver, (int(img_real_ver.shape[1]), int(img_real_ver.shape[0])))
    resized_img_real_hor = cv2.resize(img_real_hor, (int(img_real_hor.shape[1]), int(img_real_hor.shape[0])))

    gray_real_ver = cv2.cvtColor(resized_img_real_ver, cv2.COLOR_BGR2GRAY)
    gray_real_hor = cv2.cvtColor(resized_img_real_hor, cv2.COLOR_BGR2GRAY)

    gray_real_ver = cv2.equalizeHist(gray_real_ver)
    gray_real_hor = cv2.equalizeHist(gray_real_hor)

    kernel = (-1 / 256) * np.array([[1, 4, 6, 4, 1],
                                    [4, 16, 24, 6, 4],
                                    [6, 24, -476, 6, 24],
                                    [4, 16, 24, 6, 4],
                                    [1, 4, 6, 4, 1]])

    gray_real_ver = cv2.filter2D(src=gray_real_ver, ddepth=-1, kernel=kernel)
    gray_real_hor = cv2.filter2D(src=gray_real_hor, ddepth=-1, kernel=kernel)

    blur_real_ver = cv2.GaussianBlur(gray_real_ver, (5, 5), 0)
    blur_real_hor = cv2.GaussianBlur(gray_real_hor, (5, 5), 0)

    canny_real_ver = cv2.Canny(blur_real_ver, threshold1=100, threshold2=160)
    canny_real_hor = cv2.Canny(blur_real_hor, threshold1=100, threshold2=160)

    rho_ver = 1  # distance resolution in pixels of the Hough grid
    theta_ver = 1.4 * np.pi / 180  # angular resolution in radians of the Hough grid
    threshold_ver = 15  # minimum number of votes (intersections in Hough grid cell)
    min_line_length_ver = 150  # minimum number of pixels making up a line
    max_line_gap_ver = 10  # maximum gap in pixels between connectable line segments
    line_image_ver = np.copy(resized_img_real_ver) * 0  # creating a blank to draw lines on

    rho_hor = 1  # distance resolution in pixels of the Hough grid
    theta_hor = 2 * np.pi / 180  # angular resolution in radians of the Hough grid
    threshold_hor = 15  # minimum number of votes (intersections in Hough grid cell)
    min_line_length_hor = 120  # minimum number of pixels making up a line
    max_line_gap_hor = 15  # maximum gap in pixels between connectable line segments
    line_image_hor = np.copy(resized_img_real_hor) * 0  # creating a blank to draw lines on

    line_temp = np.copy(resized_img_real_hor) * 0  # creating a blank to draw lines on

    lines_ver = cv2.HoughLinesP(canny_real_ver, rho_ver, theta_ver, threshold_ver, np.array([]),
                                min_line_length_ver, max_line_gap_ver)

    lines_hor = cv2.HoughLinesP(canny_real_hor, rho_hor, theta_hor, threshold_hor, np.array([]),
                                min_line_length_hor, max_line_gap_hor)

    for line_ver in lines_ver:
        for x1, y1, x2, y2 in line_ver:
            if intended_angle_ver(x1, y1, x2, y2):
                # print(line)
                cv2.line(line_image_ver, (x1, y1), (x2, y2), (255, 0, 0), 3)

    for line_hor in lines_hor:
        for x1, y1, x2, y2 in line_hor:
            if intended_angle_hor(x1, y1, x2, y2):
                # print(line)
                cv2.line(line_image_hor, (x1, y1), (x2, y2), (255, 0, 0), 3)

    ##########

    results = []
    counter = 1

    best_result = None
    sum_len = 0

    best_x1 = None
    best_x2 = None
    best_y1 = None
    best_y2 = None
    best_x11 = None
    best_x22 = None
    best_y11 = None
    best_y22 = None
    best_x111 = None
    best_x222 = None
    best_y111 = None
    best_y222 = None

    for i in lines_ver:
        mid_x1 = None
        mid_y1 = None

        x1 = i[0][0]
        y1 = i[0][1]
        x2 = i[0][2]
        y2 = i[0][3]
        if intended_angle_ver(x1, y1, x2, y2):
            mid_x1 = (x1 + x2) / 2
            mid_y1 = (y1 + y2) / 2

        if mid_x1 is None or mid_y1 is None:
            continue
        else:
            for j in lines_ver:
                mid_x2 = None
                mid_y2 = None

                x11 = j[0][0]
                y11 = j[0][1]
                x22 = j[0][2]
                y22 = j[0][3]
                if intended_angle_ver(x11, y11, x22, y22):
                    mid_x2 = (x11 + x22) / 2
                    mid_y2 = (y11 + y22) / 2

                # if i.all() != j.all():
                #     x11 = i[0][0]
                #     y11 = i[0][1]
                #     x22 = i[0][2]
                #     y22 = i[0][3]
                #     if intended_angle_ver(x11, y11, x22, y22):
                #         mid_x2 = (x11 + x22) / 2
                #         mid_y2 = (y11 + y22) / 2

                if mid_x2 is None or mid_y2 is None:
                    continue
                else:
                    for k in lines_hor:

                        x111 = k[0][0]
                        y111 = k[0][1]
                        x222 = k[0][2]
                        y222 = k[0][3]
                        if intended_angle_hor(x111, y111, x222, y222):
                            mid_x3 = (x111 + x222) / 2
                            mid_y3 = (y111 + y222) / 2

                            if min(x1, x2) < min(x111, x222) and max(x111, x222) < max(x11, x22):
                                if min(y111, y222) > max(y1, y2, y11, y22):
                                    if check_hor_length(i, j, k):
                                        line_temp = np.copy(resized_img_real_hor) * 0
                                        for x1, y1, x2, y2 in i:
                                            cv2.line(line_temp, (x1, y1), (x2, y2), (0, 255, 0), 3)
                                        for x11, y11, x22, y22 in j:
                                            cv2.line(line_temp, (x11, y11), (x22, y22), (0, 255, 0), 3)
                                        for x111, y111, x222, y222 in k:
                                            cv2.line(line_temp, (x111, y111), (x222, y222), (0, 255, 0), 3)

                                        if best_result is None:
                                            best_result = line_temp
                                            sum_len = len_of_line(y2, y1, x2, x1) + len_of_line(y22, y11, x22,
                                                                                                x11) + len_of_line(y222,
                                                                                                                   y111,
                                                                                                                   x222,
                                                                                                                   x111)
                                            best_x1 = x1
                                            best_x2 = x2
                                            best_y1 = y1
                                            best_y2 = y2
                                            best_x11 = x11
                                            best_x22 = x22
                                            best_y11 = y11
                                            best_y22 = y22
                                            best_x111 = x111
                                            best_x222 = x222
                                            best_y111 = y111
                                            best_y222 = y222

                                        else:
                                            new_sum = len_of_line(y2, y1, x2, x1) + len_of_line(y22, y11, x22,
                                                                                                x11) + len_of_line(y222,
                                                                                                                   y111,
                                                                                                                   x222,
                                                                                                                   x111)
                                            if new_sum > sum_len:
                                                sum_len = new_sum
                                                best_result = line_temp

                                                best_x1 = x1
                                                best_x2 = x2
                                                best_y1 = y1
                                                best_y2 = y2
                                                best_x11 = x11
                                                best_x22 = x22
                                                best_y11 = y11
                                                best_y22 = y22
                                                best_x111 = x111
                                                best_x222 = x222
                                                best_y111 = y111
                                                best_y222 = y222

                                        results.append(line_temp)
                                        print(counter)
                                        counter += 1

lines_edges_hor = cv2.addWeighted(img_real_hor, 0.8, line_image_hor, 1, 0)
lines_edges_ver = cv2.addWeighted(img_real_ver, 0.8, line_image_ver, 1, 0)

lined_img = cv2.addWeighted(line_image_hor, 0.8, line_image_ver, 1, 0)
# lined_img = cv2.addWeighted(lined_img, 0.8, line_image_hor, 1, 0)

cv2.imshow(f"line result", lined_img)

# print(len(results))

# cv2.imshow(f"result", line_temp[0])
# cv2.imshow(f"result", line_temp[1])
# cv2.imshow(f"result", line_temp[2])
# cv2.imshow(f"result", line_temp[3])
# cv2.imshow(f"result", line_temp[4])

# counter = 1
# for z in results:
#     cv2.imshow(f"result {counter}", z)
#     counter += 1

cv2.imshow(f"Best Result", best_result)

cv2.line(img_real_hor, (best_x1, best_y1), (best_x2, best_y2), (0, 255, 0), 3)
cv2.line(img_real_hor, (best_x11, best_y11), (best_x22, best_y22), (0, 255, 0), 3)
cv2.line(img_real_hor, (best_x111, best_y111), (best_x222, best_y222), (0, 255, 0), 3)

cv2.imshow(f"Real Best Result", img_real_hor)

# cv2.imshow(f"line result hor", lines_edges_hor)
# cv2.imshow(f"line result ver", lines_edges_ver)

# cv2.imshow(f"Result_hor {i + 1}", lines_edges_hor)
# cv2.imshow("Result line_hor", line_image_hor)
# cv2.imshow(f"Edge_hor {i + 1}", canny_real_hor)
#
# cv2.imshow(f"Result_ver {i + 1}", lines_edges_ver)
# cv2.imshow("Result line_ver", line_image_ver)
# cv2.imshow(f"Edge_ver {i + 1}", canny_real_ver)

# path = "C:/Users/MooNice/Desktop/Edge_Detection/1/line_for_learning"
#
# cv2.imwrite(os.path.join(path, f'Result_hor {i + 1}.jpg'), lines_edges_hor)
# cv2.imwrite(os.path.join(path, f'Edge_hor {i + 1}.jpg'), canny_real_hor)

# cv2.imwrite(os.path.join(path, f'line{i + 1}.jpg'), lined_img)

# path = "C:/Users/MooNice/Desktop/Edge_Detection/1/find_pipe"
#
# cv2.imwrite(os.path.join(path, f'Result_ver {i + 1}.jpg'), lines_edges_ver)
# cv2.imwrite(os.path.join(path, f'Edge_ver {i + 1}.jpg'), canny_real_ver)

cv2.waitKey(0)
cv2.destroyAllWindows()
