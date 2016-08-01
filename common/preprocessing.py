#
# Common functions
#

# Python
import os
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


def read_image(filename, size):
    assert os.path.exists(filename), "Image path '%s' from file list is not found" % filename
    img = cv2.imread(filename)
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if size is not None:
        rgb = cv2.resize(rgb, (size[0], size[1]))
    return rgb


# def get_raw_data(files, width, height):
#     out = None
#     for i, f in enumerate(files):
#         data = read_image(f, (width, height))
#         if out is None:
#             out = np.empty((len(files), ))
#
#         out[i, :, :, :] = data[:, :, :]
#     return out


def get_data2(files, width, height):
    out = np.empty((len(files), width * height))
    for i, f in enumerate(files):
        image = cv2.imread(f)
        data = preprocess_image(image, width, height)
        out[i, :] = data[:]
    return out


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


def get_data_parallel2(files, width, height):
    # define output image shape from the first image:
    n_workers = multiprocessing.cpu_count()
    pool = multiprocessing.Pool(processes=n_workers)
    out = pool.map(_process_file, itertools.izip(files, itertools.repeat((width, height))))
    pool.close()
    return np.array(out)


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
