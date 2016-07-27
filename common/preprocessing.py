#
# Common functions
#

# Python
import multiprocessing
import itertools


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


def _process_file(args):
    f = args[0]
    size = args[1]
    image = cv2.imread(f)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    gray = cv2.resize(gray, size)
    return gray.reshape(size[0]*size[1])


def get_data_parallel(files, resize_factor):

    # define output image shape from the first image:
    def _define_shape(f, factor):
        image = cv2.imread(f)
        return image.shape[1] / factor, image.shape[0] / factor

    w, h = _define_shape(files[0], resize_factor)

    n_workers = multiprocessing.cpu_count()
    pool = multiprocessing.Pool(processes=n_workers)
    out = pool.map(_process_file, itertools.izip(files, itertools.repeat((w, h))))
    pool.close()

    return np.array(out), w, h
