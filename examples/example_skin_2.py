
#
# Example : apply opencv skin detector
#

import os
import time

import numpy as np

import matplotlib.pyplot as plt

import cv2


# Implementation of http://www.bytefish.de/blog/opencv/skin_color_thresholding/
def rule_1(rgb_image):
    """
    Rule one to segment the skin on an rgb image
    Output is bool matrix of input image size
    """
    out = np.zeros((rgb_image.shape[0], rgb_image.shape[1]), dtype=np.bool)

    def process_line(r, g, b):
        rg_abs_diff = np.abs(r.astype(np.float) - g.astype(np.float))
        e0 = r > b

        _and = np.logical_and
        # e1 = (r > 95) and (g > 40) and (b > 20) and \
        #      ((max(r, max(g, b)) - min(r, min(g, b))) > 15) and \
        #      (abs(r-g) > 15) and (r > g) and (r > b)
        e1 = _and(r > 95, np.logical_and(g > 40, b > 20))
        e1 = _and(e1, np.maximum(r, np.maximum(g, b)) - np.minimum(r, np.minimum(g, b)) > 15)
        e1 = _and(e1, _and(rg_abs_diff > 15, _and(r > g, e0)))
        # e2 = (r > 220) and (g > 210) and (b > 170) and \
        #      (abs(r-g) <= 15) and (r > b) and (g > b)
        e2 = _and(r > 220, _and(g > 210, b > 170))
        e2 = _and(e2, _and(rg_abs_diff <= 15, _and(e0, g > b)))
        return np.logical_or(e1, e2)

    for i in range(out.shape[0]):
        out[i, :] = process_line(rgb_image[i, :, 2], rgb_image[i, :, 1], rgb_image[i, :, 0])
    return out


def rule_2(ycrcb_image):
    """
    Rule one to segment the skin on an y_cr_cb image (dtype=np.float)
    Output is bool matrix of input image size
    """
    out = np.zeros((ycrcb_image.shape[0], ycrcb_image.shape[1]), dtype=np.bool)

    def process_line(y, cr, cb):
        e3 = cr <= 1.5862*cb + 20
        e4 = cr >= 0.3448*cb + 76.2069
        e5 = cr >= -4.5652*cb + 234.5652
        e6 = cr <= -1.15*cb + 301.75
        e7 = cr <= -2.2857*cb + 432.85

        _and = np.logical_and
        return _and(e3, _and(e4, _and(e5, _and(e6, e7))))

    for i in range(out.shape[0]):
        out[i, :] = process_line(ycrcb_image[i, :, 0], ycrcb_image[i, :, 1], ycrcb_image[i, :, 2])
    return out


def rule_3(hsv_image):
    """
    Rule one to segment the skin on an hsv image
    Output is bool matrix of input image size
    """
    out = np.zeros((hsv_image.shape[0], hsv_image.shape[1]), dtype=np.bool)

    def process_line(h, s, v):
        # Original
        # return 255*np.logical_or(h < 25, h > 230)
        # Mine
        # skin_range_min = np.array([6, 10, 35], dtype=np.uint8)
        # skin_range_max = np.array([45, 255, 255], dtype=np.uint8)
        _and = np.logical_and
        e1 = _and(h > 6, h < 45)
        e2 = _and(s > 10, s < 255)
        e3 = _and(v > 35, v < 255)
        return _and(e1, _and(e2, e3))

    for i in range(out.shape[0]):
        out[i, :] = process_line(hsv_image[i, :, 0], hsv_image[i, :, 1], hsv_image[i, :, 2])
    return out


def get_skin(image):

    image_ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCR_CB).astype(np.float)
    image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # image_hsv = cv2.normalize(image_hsv, alpha=0.0, beta=255.0, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32FC3)

    m1 = rule_1(image)
    m2 = rule_2(image_ycrcb)
    m3 = rule_3(image_hsv)

    res = 255*np.logical_and(m1, np.logical_and(m2, m3)).astype(dtype=np.uint8)
    return res, [m1, m2, m3]



train_files_path = "../input/train/c0"
# train_files_path = "../input/test"

counter = 10
for f in os.listdir(train_files_path):

    image = cv2.imread(os.path.join(train_files_path, f))

    # Start image preprocessing:
    start = time.time()

    proc = cv2.resize(image, (0, 0), fx=0.5, fy=0.5, interpolation=cv2.INTER_LINEAR)
    # proc = cv2.medianBlur(proc, 7)

    print proc.shape
    ### Detect skin
    skin_mask, masks = get_skin(proc)

    # Filter the skin mask :
    # skin_mask = sieve(skin_like_mask, skin_sieve_min_size)
    # kernel = np.ones((skin_kernel_size, skin_kernel_size), dtype=np.int8)
    # skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_CLOSE, kernel)

    # Apply skin mask
    # skin_segm_rgb = cv2.bitwise_and(image, image, mask=skin_mask)

    print "Elapsed seconds : ", time.time() - start

    # Show the output image :
    cv2.imshow("Original", proc)
    cv2.imshow("Skin Mask", skin_mask)
    cv2.imshow("Skin Mask 1", 255*masks[0].astype(np.uint8))
    cv2.imshow("Skin Mask 2", 255*masks[1].astype(np.uint8))
    cv2.imshow("Skin Mask 3", 255*masks[2].astype(np.uint8))
    # cv2.imshow("Skin segmentation", skin_segm_rgb)
    cv2.waitKey(0)

    counter -= 1
    if counter == 0:
        break

