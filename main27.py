import math

import cv2
import os
import numpy as np


def intended_angle_ver(x_1, y_1, x_2, y_2):
    delta_y = float(y_2 - y_1)
    delta_x = float(x_2 - x_1)

    try:
        angle_degrees_ver = np.arctan(delta_y / delta_x) * 180.0 / np.pi
    except:
        return False

    angle_degrees_ver = abs(angle_degrees_ver)

    # print('--------------')
    # print(angle_degrees_hor)

    if (angle_degrees_ver < 160.0 and angle_degrees_ver > 20.0):
        return True
    return False


def intended_angle_hor(x_1, y_1, x_2, y_2):
    delta_y = float(y_2 - y_1)
    delta_x = float(x_2 - x_1)

    try:
        angle_degrees_hor = np.arctan(delta_y / delta_x) * 180.0 / np.pi
    except:
        return False

    angle_degrees_hor = abs(angle_degrees_hor)

    # print('--------------')
    # print(angle_degrees_ver)

    if (angle_degrees_hor < 15.0):  # angle_degrees_hor < 30
        return True
    return False


def line_coefficient(x_1, y_1, x_2, y_2):
    a = float((y_2 - y_1) / (x_2 - x_1))
    b = float(y_1 - (a * x_1))

    return a, b


def collision_point(a_1, b_1, a_2, b_2):
    x = float((b_2 - b_1) / (a_1 - a_2))
    y = float(a_1 * x + b_1)

    # print(f"x: {x}")
    # print(f"y: {y}")

    return x, y


def out_of_range(x, y):
    # return True
    return 0 <= x <= int(resized_img_real_ver.shape[1]) and 0 <= y <= int(resized_img_real_ver.shape[0])


def len_of_line(y_2, y_1, x_2, x_1):
    return float(math.sqrt((y_2 - y_1) ** 2 + (x_2 - x_1) ** 2))


def two_line_intersection_angle(m, mm):
    tetha = math.atan(float(math.fabs((m - mm) / (1 + m * mm))) * (180.0 / math.pi))
    return tetha


def check_hor_length(i, j, k):
    # return True
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

    if out_of_range(col_x_1_3, col_y_1_3) and out_of_range(col_x_2_3, col_y_2_3):
        # print(col_x_1_3, col_y_1_3)
        # print(col_x_2_3, col_y_2_3)

        hor_len = len_of_line(col_y_2_3, col_y_1_3, col_x_2_3, col_x_1_3)

        # print(hor_len)
        # print('---------------')
        # return 300.0 < hor_len < 600.0

        avg_mid_hor = (y111 + y222) / 2

        if avg_mid_hor >= 550:
            return 490.0 - 50.0 < hor_len < 490.0 + 50.0
        elif avg_mid_hor >= 450:
            return 410.0 - 50.0 < hor_len < 410.0 + 50.0
        elif avg_mid_hor >= 350:
            return 350.0 - 50.0 < hor_len < 350.0 + 50.0
        else:
            return 310.0 - 30.0 < hor_len < 310 + 30.0

    return False


def check_collision_point_placement(x1, x2, y1, y2, x11, x22, y11, y22, x111, x222, y111, y222):
    # return True
    a1, b1 = line_coefficient(x1, y1, x2, y2)
    a2, b2 = line_coefficient(x11, y11, x22, y22)
    a3, b3 = line_coefficient(x111, y111, x222, y222)

    col_x_1_3, col_y_1_3 = collision_point(a1, b1, a3, b3)
    col_x_2_3, col_y_2_3 = collision_point(a2, b2, a3, b3)

    # print(1)

    # if (col_x_1_3 > 0.0 and col_x_1_3 < min(x1, x2, x111, x222)) and (
    #         col_x_2_3 < float(resized_img_real_ver.shape[1]) and col_x_2_3 > max(x11, x22, x111, x222)):
    #     return True

    if col_x_1_3 > 0.0 and col_x_2_3 < float(resized_img_real_ver.shape[1]):
        return True

    # try:
    #     pass
    #     # print("--------------")
    #     # print(f"a1: {a1}")
    #     # print(f"b1: {b1}")
    #     # print(f"a2: {a2}")
    #     # print(f"b2: {b2}")
    #     # col_x_1_2, col_y_1_2 = collision_point(a1, b1, a1, b2)
    #     # print(col_x_1_2, col_y_1_2)
    #
    #     # if (max(x1, x2) < col_x_1_2 < min(x11, x22)) and (col_y_1_2 < min(y1, y2, y11, y22)):
    #     #     return True
    # except:
    #     # print(2)
    #     return True
    return False


if __name__ == '__main__':
    MID_MARGIN = 150  # 150
    IMG_NUMBERS = 1  # 2186

    # for index in range(2186):
    for index in range(IMG_NUMBERS):
        global resized_img_real_ver
        # image_name = f'frame {index + 1}.jpg'
        image_name = f'frame {96}.jpg'
        IMG_PATH_REAL = os.path.join('new_crop_frames', image_name)
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

        try:
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
        except:
            continue

        ##########

        results = []
        # counter = 1

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
            else:
                continue

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

                    if x1 == x11 and x2 == x22 and y1 == y11 and y2 == y22:
                        continue

                    if intended_angle_ver(x11, y11, x22, y22):
                        mid_x2 = (x11 + x22) / 2
                        mid_y2 = (y11 + y22) / 2
                    else:
                        continue

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
                            else:
                                continue

                            if mid_x3 is None or mid_y3 is None:
                                continue
                            else:

                                # if (min(x1, x2) < min(x111, x222) and max(x111, x222) < max(x11, x22)) and (
                                #         ((mid_x1 + mid_x2) / 2) - MID_MARGIN < mid_x3 < (mid_x1 + mid_x2) + MID_MARGIN):
                                if ((mid_x1 + mid_x2) / 2) - MID_MARGIN < mid_x3 < ((mid_x1 + mid_x2) / 2) + MID_MARGIN:
                                    if min(y111, y222) > max(y1, y2, y11, y22):
                                        if check_hor_length(i, j, k):
                                            if check_collision_point_placement(x1, x2, y1, y2, x11, x22, y11, y22, x111,
                                                                               x222, y111, y222):
                                                # m, not_m = line_coefficient(x1, y1, x2, y2)
                                                # mm, not_mm = line_coefficient(x11, y11, x22, y22)
                                                # mmm, not_mmm = line_coefficient(x111, y111, x222, y222)
                                                # print('==========')
                                                # print((mid_x1 + mid_x2) / 2)

                                                # print(m)
                                                # print(mm)
                                                # print(mmm)
                                                # print('---')
                                                # print(two_line_intersection_angle(m, mmm))
                                                # print(two_line_intersection_angle(mm, mmm))
                                                # print('---------------------')
                                                # if two_line_intersection_angle(m, mmm) < 90 and two_line_intersection_angle(mm,
                                                #                                                                             mmm) > 90:
                                                line_temp = np.copy(resized_img_real_hor) * 0
                                                # for x1, y1, x2, y2 in i:
                                                #     cv2.line(line_temp, (x1, y1), (x2, y2), (0, 255, 0), 3)
                                                # for x11, y11, x22, y22 in j:
                                                #     cv2.line(line_temp, (x11, y11), (x22, y22), (0, 255, 0), 3)
                                                # for x111, y111, x222, y222 in k:
                                                #     cv2.line(line_temp, (x111, y111), (x222, y222), (0, 255, 0), 3)
                                                cv2.line(line_temp, (x1, y1), (x2, y2), (255, 0, 0), 3)
                                                cv2.line(line_temp, (x11, y11), (x22, y22), (0, 255, 0), 3)
                                                cv2.line(line_temp, (x111, y111), (x222, y222), (0, 0, 255), 3)

                                                results.append(line_temp)

                                                # cv2.imshow('result one in', line_temp)

                                                if best_result is None:
                                                    best_result = line_temp
                                                    # sum_len = len_of_line(y2, y1, x2, x1) + len_of_line(y22, y11, x22,
                                                    #                                                     x11) + len_of_line(y222,
                                                    #                                                                        y111,
                                                    #                                                                        x222,
                                                    #                                                                        x111)

                                                    sum_len = len_of_line(y2, y1, x2, x1) + len_of_line(y22, y11, x22,
                                                                                                        x11)
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
                                                    # new_sum = len_of_line(y2, y1, x2, x1) + len_of_line(y22, y11, x22,
                                                    #                                                     x11) + len_of_line(y222,
                                                    #                                                                        y111,
                                                    #                                                                        x222,
                                                    #                                                                        x111)

                                                    new_sum = len_of_line(y2, y1, x2, x1) + len_of_line(y22, y11, x22,
                                                                                                        x11)
                                                    if new_sum == sum_len:
                                                        a_111, b111 = line_coefficient(x111, y111, x222, y222)
                                                        a_1, b1 = line_coefficient(best_x111, best_y111, best_x222,
                                                                                   best_y222)

                                                        if math.fabs(a_111) < math.fabs(a_1):
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

                                                    elif new_sum > sum_len:
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

                                                if len(results) == 8:
                                                    print('start')
                                                # print(counter)
                                                # counter += 1

    cv2.line(img_real_hor, (best_x1, best_y1), (best_x2, best_y2), (255, 0, 0), 3)
    cv2.line(img_real_hor, (best_x11, best_y11), (best_x22, best_y22), (0, 255, 0), 3)
    cv2.line(img_real_hor, (best_x111, best_y111), (best_x222, best_y222), (0, 0, 255), 3)

    a__1, b__1 = line_coefficient(best_x1, best_y1, best_x2, best_y2)
    a__2, b__2 = line_coefficient(best_x11, best_y11, best_x22, best_y22)
    a__3, b__3 = line_coefficient(best_x111, best_y111, best_x222, best_y222)

    col_x_1_3_best, col_y_1_3_best = collision_point(a__1, b__1, a__3, b__3)
    col_x_2_3_best, col_y_2_3_best = collision_point(a__2, b__2, a__3, b__3)

    print('--------------------------')
    print(best_y111, best_y222)
    print(len_of_line(col_y_2_3_best, col_y_1_3_best, col_x_2_3_best, col_x_1_3_best))

    print()

    cv2.imshow(f"Real Best Result", img_real_hor)

    # path = "C:/Users/MooNice/Desktop/Edge_Detection/1/finding_pip/find_exact_pipe_2"
    # print(index + 1)
    # cv2.imwrite(os.path.join(path, f'Result {index + 1}.jpg'), img_real_hor)

    ind = 1
    for a in results:
        cv2.imshow(f" result {ind}", a)
        ind += 1
#
# cv2.imshow(f"best result", img_real_hor)

# cv2.imshow(f"edge hor ", canny_real_hor)
# cv2.imshow(f"edge ver", canny_real_ver)

# lines_edges_hor = cv2.addWeighted(img_real_hor, 0.8, line_image_hor, 1, 0)
# lines_edges_ver = cv2.addWeighted(img_real_ver, 0.8, line_image_ver, 1, 0)

# lined_img = cv2.addWeighted(line_image_hor, 0.8, line_image_ver, 1, 0)
# lined_img = cv2.addWeighted(lined_img, 0.8, line_image_hor, 1, 0)

# cv2.imshow(f"line result", lined_img)

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

# cv2.imshow(f"Best Result", best_result)

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
