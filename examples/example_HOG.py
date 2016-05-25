
#
# Example : apply opencv HOG people detector
#

import os
import time

import numpy as np

import matplotlib.pyplot as plt

import cv2

win_stride = (8, 8)
padding = (32, 32)
scale = 1.05
use_meanshift = True

hog = cv2.HOGDescriptor((64, 128), (16, 16), (8, 8), (8, 8), 9)
print "HOG descriptor size : ", hog.getDescriptorSize()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())


train_files_path = "../input/train/c1"
counter = 1
for f in os.listdir(train_files_path):

    image = cv2.imread(os.path.join(train_files_path, f))
    image = cv2.resize(image, (0, 0), fx=0.7, fy=0.7, interpolation=cv2.INTER_LINEAR)


    # Compute gradients :
    # grads, angles = hog.computeGradient(image)
    # # Show the output image :
    # cv2.imshow("Gradient", grads)
    # cv2.imshow("Angles", angles)
    # cv2.waitKey(0)


    start = time.time()
    rects, weights = hog.detectMultiScale(image,
                                          winStride=win_stride,
                                          padding=padding,
                                          scale=scale)
    print "Elapsed seconds : ", time.time() - start
    print "Nb rects : ", len(rects)

    # Draw rectangles :
    for (x, y, w, h) in rects:
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Show the output image :
    cv2.imshow("Detections", image)
    cv2.waitKey(0)


    if counter < 0:
        break
    counter -= 1