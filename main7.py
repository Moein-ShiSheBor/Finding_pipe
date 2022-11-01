import cv2
import os

IMG_PATH_REAL = os.path.join('IMG', 'full.jpg')
img_real = cv2.imread(IMG_PATH_REAL)

num = 0
for i in range(3):
    resize = img_real[i * 60:min(i * 60 + 440, 558), 0: 992 ]
    print(resize.shape)
    cv2.imshow(f"Pic {num + 1}", resize)
    num += 1

cv2.waitKey(0)
cv2.destroyAllWindows()
