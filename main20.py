import cv2
import os
import numpy as np


def intended_angle_hor(x1, y1, x2, y2):
    delta_y = y2 - y1
    delta_x = x2 - x1

    angle_degrees_hor = np.arctan(delta_y / delta_x) * 180 / np.pi

    angle_degrees_hor = abs(angle_degrees_hor)

    # print('--------------')
    # print(angle_degrees_hor)

    if (angle_degrees_hor < 160 and angle_degrees_hor > 20):
        return True
    return False


def intended_angle_ver(x1, y1, x2, y2):
    delta_y = y2 - y1
    delta_x = x2 - x1

    angle_degrees_ver = np.arctan(delta_y / delta_x) * 180 / np.pi

    angle_degrees_ver = abs(angle_degrees_ver)

    # print('--------------')
    # print(angle_degrees_ver)

    if (angle_degrees_ver < 30):
        return True
    return False


for i in range(1):
    image_name = f'frame38-1.jpg'
    # image_name = f'frame35.jpg'
    IMG_PATH_REAL = os.path.join('test_frame', image_name)
    img_real_hor = cv2.imread(IMG_PATH_REAL)
    img_real_ver = cv2.imread(IMG_PATH_REAL)

    resized_img_real_hor = cv2.resize(img_real_hor, (int(img_real_hor.shape[1]), int(img_real_hor.shape[0])))
    resized_img_real_ver = cv2.resize(img_real_ver, (int(img_real_ver.shape[1]), int(img_real_ver.shape[0])))

    gray_real_hor = cv2.cvtColor(resized_img_real_hor, cv2.COLOR_BGR2GRAY)
    gray_real_ver = cv2.cvtColor(resized_img_real_ver, cv2.COLOR_BGR2GRAY)

    gray_real_hor = cv2.equalizeHist(gray_real_hor)
    gray_real_ver = cv2.equalizeHist(gray_real_ver)

    kernel = (-1 / 256) * np.array([[1, 4, 6, 4, 1],
                                    [4, 16, 24, 6, 4],
                                    [6, 24, -476, 6, 24],
                                    [4, 16, 24, 6, 4],
                                    [1, 4, 6, 4, 1]])

    gray_real_hor = cv2.filter2D(src=gray_real_hor, ddepth=-1, kernel=kernel)
    gray_real_ver = cv2.filter2D(src=gray_real_ver, ddepth=-1, kernel=kernel)

    blur_real_hor = cv2.GaussianBlur(gray_real_hor, (5, 5), 0)
    blur_real_ver = cv2.GaussianBlur(gray_real_ver, (5, 5), 0)

    canny_real_hor = cv2.Canny(blur_real_hor, threshold1=100, threshold2=160)
    canny_real_ver = cv2.Canny(blur_real_ver, threshold1=100, threshold2=160)

    rho_hor = 1  # distance resolution in pixels of the Hough grid
    theta_hor = 1.4 * np.pi / 180  # angular resolution in radians of the Hough grid
    threshold_hor = 15  # minimum number of votes (intersections in Hough grid cell)
    min_line_length_hor = 150  # minimum number of pixels making up a line
    max_line_gap_hor = 10  # maximum gap in pixels between connectable line segments
    line_image_hor = np.copy(resized_img_real_hor) * 0  # creating a blank to draw lines on

    rho_ver = 1  # distance resolution in pixels of the Hough grid
    theta_ver = 2 * np.pi / 180  # angular resolution in radians of the Hough grid
    threshold_ver = 15  # minimum number of votes (intersections in Hough grid cell)
    min_line_length_ver = 120  # minimum number of pixels making up a line
    max_line_gap_ver = 15  # maximum gap in pixels between connectable line segments
    line_image_ver = np.copy(resized_img_real_ver) * 0  # creating a blank to draw lines on

    line_temp = np.copy(resized_img_real_hor) * 0  # creating a blank to draw lines on

    lines_hor = cv2.HoughLinesP(canny_real_hor, rho_hor, theta_hor, threshold_hor, np.array([]),
                                min_line_length_hor, max_line_gap_hor)

    lines_ver = cv2.HoughLinesP(canny_real_ver, rho_ver, theta_ver, threshold_ver, np.array([]),
                                min_line_length_ver, max_line_gap_ver)

    for line_hor in lines_hor:
        for x1, y1, x2, y2 in line_hor:
            if intended_angle_hor(x1, y1, x2, y2):
                # print(line)
                cv2.line(line_image_hor, (x1, y1), (x2, y2), (255, 0, 0), 3)

    for line_ver in lines_ver:
        for x1, y1, x2, y2 in line_ver:
            if intended_angle_ver(x1, y1, x2, y2):
                # print(line)
                cv2.line(line_image_ver, (x1, y1), (x2, y2), (255, 0, 0), 3)

    ##########

    results = []

    for i in lines_hor:
        for x1, y1, x2, y2 in i:
            if intended_angle_hor(x1, y1, x2, y2):
                mid_x1 = (x1 + x2) / 2
                mid_y1 = (y1 + y2) / 2
            else:
                break

        for j in lines_hor:

            for x1, y1, x2, y2 in j:
                if intended_angle_hor(x1, y1, x2, y2):
                    mid_x2 = (x1 + x2) / 2
                    mid_y2 = (y1 + y2) / 2
                else:
                    break

            for k in lines_ver:
                for x1, y1, x2, y2 in k:
                    if intended_angle_ver(x1, y1, x2, y2):
                        mid_x3 = (x1 + x2) / 2
                        mid_y3 = (y1 + y2) / 2
                    else:
                        break
                    try:
                        if (mid_x1 < mid_x3 < mid_x2) or (mid_x1 > mid_x3 > mid_x2):
                            if mid_y3 < mid_y1 and mid_y3 < mid_y2:
                                line_temp = np.copy(resized_img_real_hor) * 0  # creating a blank to draw lines on
                                for x1, y1, x2, y2 in i:
                                    cv2.line(line_temp, (x1, y1), (x2, y2), (0, 255, 0), 3)
                                for x1, y1, x2, y2 in j:
                                    cv2.line(line_temp, (x1, y1), (x2, y2), (0, 255, 0), 3)
                                for x1, y1, x2, y2 in k:
                                    cv2.line(line_temp, (x1, y1), (x2, y2), (0, 255, 0), 3)

                                # cv2.imshow(f"result", line_temp)
                                results.append(line_temp)
                                print(1)
                                break
                    except:
                        pass

    lines_edges_hor = cv2.addWeighted(img_real_hor, 0.8, line_image_hor, 1, 0)
    lines_edges_ver = cv2.addWeighted(img_real_ver, 0.8, line_image_ver, 1, 0)

    lined_img = cv2.addWeighted(line_image_hor, 0.8, line_image_ver, 1, 0)
    # lined_img = cv2.addWeighted(lined_img, 0.8, line_image_hor, 1, 0)

    cv2.imshow(f"line result", lined_img)

    print(len(results))

    # cv2.imshow(f"result", line_temp[0])
    # cv2.imshow(f"result", line_temp[1])
    # cv2.imshow(f"result", line_temp[2])
    # cv2.imshow(f"result", line_temp[3])
    # cv2.imshow(f"result", line_temp[4])

    counter = 1
    for z in results:
        cv2.imshow(f"result {counter}", z)
        counter += 1

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
