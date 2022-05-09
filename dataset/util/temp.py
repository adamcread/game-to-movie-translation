import os
import numpy as np
import cv2 as cv

images = set(os.listdir('../frames/train/trainB/'))
masks = set(map(lambda x: x.replace('.png', '.jpg'), os.listdir('../frames/mask/trainB/')))

for image in images.difference(masks):
    frame = cv.imread('../frames/train/trainB/'+image)
    blank = np.zeros(frame.shape)

    cv.imwrite('../frames/mask/trainB/'+image.replace('.jpg', '.png'), blank)
