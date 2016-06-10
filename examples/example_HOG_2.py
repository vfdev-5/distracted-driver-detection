
#
# Example : apply opencv HOG people detector
#

import os
import time

import numpy as np

import matplotlib.pyplot as plt

import cv2

loc = []


def foo(_loc):
    _loc.append(1)

foo(loc)
print loc


exit(1)

hog = cv2.HOGDescriptor()
print "HOG descriptor size : ", hog.getDescriptorSize()


train_files_path = "../input/train/c1"
counter = 1
for f in os.listdir(train_files_path):

    image = cv2.imread(os.path.join(train_files_path, f))
    image = cv2.resize(image, (0, 0), fx=0.5, fy=0.5, interpolation=cv2.INTER_LINEAR)

    print image.shape

    # Compute descriptors
    features = hog.compute(image, winStride=(8, 8), padding=(0, 0))

    # Compute gradients :
    # grads, angles = hog.computeGradient(image)
    # print grads.shape, grads.dtype, angles.shape, angles.dtype
    # Show the output image :
    # abs_grad = cv2.magnitude(grads[:, :, 0], grads[:, :, 1])
    # cv2.imshow("Abs gradient", abs_grad)
    # cv2.imshow("Angles 1", angles[:, :, 0])
    # cv2.imshow("Angles 2", angles[:, :, 1])
    # cv2.waitKey(0)
    # plt.figure()
    # plt.title("Abs gradient")
    # plt.imshow(abs_grad)
    # plt.figure()
    # plt.title("Angles 1")
    # plt.imshow(angles[:, :, 0])
    # plt.figure()
    # plt.title("Angles 2")
    # plt.imshow(angles[:, :, 1])
    #
    # plt.show()

    # start = time.time()
    # rects, weights = hog.detectMultiScale(image,
    #                                       winStride=win_stride,
    #                                       padding=padding,
    #                                       scale=scale)
    # print "Elapsed seconds : ", time.time() - start
    # print "Nb rects : ", len(rects)
    #
    # # Draw rectangles :
    # for (x, y, w, h) in rects:
    #     cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
    #
    # # Show the output image :
    # cv2.imshow("Detections", image)
    # cv2.waitKey(0)


    counter -= 1
    if counter == 0:
        break
