#
# Common functions
#

# numpy
import numpy as np

# Opencv
import cv2


def preprocess_image(in_image, w, h):
    gray = cv2.cvtColor(in_image, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    gray = cv2.resize(gray, (w, h))
    return gray.reshape(w*h)


def get_data(files, resize_factor):
    out = None
    w = None
    h = None
    for i, f in enumerate(files):
        image = cv2.imread(f)
        if out is None:
            w, h = image.shape[1] / resize_factor, image.shape[0] / resize_factor
            out = np.empty((len(files), w * h))
        data = preprocess_image(image, w, h)
        out[i, :] = data[:]
    return out, w, h
