import cv2, os

for i in range(2187):
    image_name = f'frame{i + 1}.jpg'
    IMG_PATH_REAL = os.path.join('frame_save_crop', image_name)
    img = cv2.imread(IMG_PATH_REAL)

    # print(int(img.shape[1]), int(img.shape[0]))

    y = 0
    x = 0
    h = 1184
    w = 665

    crop_image = img[x:w, y:h]
    path = "C:/Users/MooNice/Desktop/Edge_Detection/1/new_crop_frames"

    cv2.imwrite(os.path.join(path, f'frame{i + 1}.jpg'), crop_image)
    # cv2.imshow("Cropped", crop_image)
cv2.waitKey(0)
