#
# Common functions
#

# Python
import os
import platform
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
    """
    Read image from file
    :param filename: path to image
    :param size: list of width and height, e.g [100, 120]
    :return: ndarray with image data. ndarray.shape = (height, widht, number of channels)
    """
    assert os.path.exists(filename), "Image path '%s' from file list is not found" % filename
    assert len(size) == 2 and size[0] > 0 and size[1] > 0, "Size should be a list of 2 positive values" 
    img = cv2.imread(filename) # img.shape = (H, W, Ch)
    if len(img.shape) == 3 and img.shape[2] == 3: # if image has 3 channels
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    elif len(img.shape) == 2:
        img = img.reshape(img.shape + (1,))
    if size is not None:
        img = cv2.resize(img, (size[0], size[1]))
    return img
    

def _get_raw_data(files, width, height):
    out = None
    for i, f in enumerate(files):
        data = read_image(f, (width, height))
        if out is None:
            k = data.shape[2]
            out = np.empty((len(files), k, height, width), dtype=np.uint8)
        out[i, :, :, :] = data[:, :, :].T
    return out


def _open_file(args):
    f = args[0]
    size = args[1]
    data = read_image(f, (size[0], size[1]))
    return data


def _get_raw_data_parallel(files, width, height):
    # define output image shape from the first image:
    n_workers = multiprocessing.cpu_count()
    pool = multiprocessing.Pool(processes=n_workers)
    out = pool.map(_open_file, itertools.izip(files, itertools.repeat((width, height))))
    pool.close()
    out = np.array(out)
    out = out.transpose((0, 3, 2, 1))
    return out


def get_raw_data(files, width, height):
    
    # Do not use multiprocessing on MacOSX
    if "Darwin" not in platform.system():
        return _get_raw_data_parallel(files, width, height)
    else:
        return _get_raw_data(files, width, height)


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
