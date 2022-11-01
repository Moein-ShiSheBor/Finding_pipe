import os
import os

import cv2
import numpy as np
import glob

frameSize = (1184, 665)

out = cv2.VideoWriter('sample_finding_pipe_3.avi', cv2.VideoWriter_fourcc(*'DIVX'), 15, frameSize)

for i in range(2186):
    image_name = f'result{i + 1}.jpg'
    IMG_PATH_REAL = os.path.join('finding_pip/find_exact_pipe_5', image_name)
    img = cv2.imread(IMG_PATH_REAL)
    out.write(img)

out.release()
