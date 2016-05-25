
#
# Example : apply opencv bag of words
#

import os
import time

import numpy as np

import cv2

train_files_path = "../input/train/c0"

counter = 2
for f in os.listdir(train_files_path):

    image = cv2.imread(os.path.join(train_files_path, f))

    # Start image preprocessing:
    start = time.time()

    image = cv2.resize(image, (0, 0), fx=0.5, fy=0.5, interpolation=cv2.INTER_LINEAR)
    proc = cv2.medianBlur(image, 7)

    ### Detect skin
    image_hsv = cv2.cvtColor(proc, cv2.COLOR_BGR2HSV)
    skin_like_mask = cv2.inRange(image_hsv, skin_range_min, skin_range_max)
    # Filter the skin mask :
    skin_mask = sieve(skin_like_mask, skin_sieve_min_size)
    kernel = np.ones((skin_kernel_size,skin_kernel_size), dtype=np.int8)
    skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_CLOSE, kernel)
    # Apply skin mask
    skin_segm_rgb = cv2.bitwise_and(image, image, mask=skin_mask)


    ### Contours
    proc = cv2.Canny(proc, wheel_canny_threshold,
                     wheel_canny_ratio*wheel_canny_threshold)
    cv2.imshow("Proc", proc)

    contours = proc
    contours_rgb = np.zeros(contours.shape + (3,), dtype=np.uint8)
    contours_rgb[:, :, 0] = 0
    contours_rgb[:, :, 1] = contours
    contours_rgb[:, :, 2] = 0

    ## Resulting image:
    result = skin_segm_rgb + contours_rgb

    print "Elapsed seconds : ", time.time() - start

    # Show the output image :
    cv2.imshow("Original", image)
    cv2.imshow("Mask", skin_mask)
    cv2.imshow("Result", result)
    cv2.waitKey(0)

    if counter <= 0:
        break
    counter -= 1

