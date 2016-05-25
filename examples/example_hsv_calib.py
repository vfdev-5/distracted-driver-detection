
#
# Example : hsv range calibration
#

import os
import time

import numpy as np

import cv2

pallette_hs = np.empty((256, 180, 3), dtype=np.uint8)
pallette_hv = np.empty((256, 180, 3), dtype=np.uint8)
h = np.arange(0, 180, dtype=np.uint8)
t = np.arange(0, 256, dtype=np.uint8)

for i in t:
    pallette_hs[i, :, 0] = h[:]
    pallette_hs[i, :, 1] = i
    pallette_hs[i, :, 2] = 150
    pallette_hv[i, :, 0] = h[:]
    pallette_hv[i, :, 1] = 150
    pallette_hv[i, :, 2] = i


def func(pallette, rmin, rmax):
    # in HSV :
    mask = cv2.inRange(pallette, rmin, rmax)
    # Apply mask :
    return cv2.bitwise_and(pallette, pallette, mask=mask)

skin_range_min = np.array([6, 0, 80], dtype=np.uint8)
skin_range_max = np.array([45, 255, 255], dtype=np.uint8)


pallette_hs = func(pallette_hs, skin_range_min, skin_range_max)
pallette_hv = func(pallette_hv, skin_range_min, skin_range_max)

pallette_rgb_1 = cv2.cvtColor(pallette_hs, cv2.COLOR_HSV2BGR)
pallette_rgb_1 = cv2.resize(pallette_rgb_1, (0, 0), fx=2, fy=2, interpolation=cv2.INTER_NEAREST)
pallette_rgb_2 = cv2.cvtColor(pallette_hv, cv2.COLOR_HSV2BGR)
pallette_rgb_2 = cv2.resize(pallette_rgb_2, (0, 0), fx=2, fy=2, interpolation=cv2.INTER_NEAREST)

# Show the output image :
cv2.imshow("pallette rgb : hue-sat", pallette_rgb_1)
cv2.imshow("pallette rgb : hue-val", pallette_rgb_2)
cv2.waitKey(0)
