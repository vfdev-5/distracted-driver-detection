
#
# Example : create opencv bag of words
#

import os
import time

import numpy as np

import cv2

train_files_path = "../input/train/c1"

counter = 5
for f in os.listdir(train_files_path):

    image = cv2.imread(os.path.join(train_files_path, f))

    # Start image preprocessing:
    start = time.time()
    image = cv2.resize(image, (0, 0), fx=0.5, fy=0.5, interpolation=cv2.INTER_LINEAR)
    proc = cv2.medianBlur(image, 7)



    print "Elapsed seconds : ", time.time() - start

    # Show the output image :
    cv2.imshow("Original", image)
    cv2.waitKey(0)

    if counter <= 0:
        break
    counter -= 1

