#!/bin/python2

# Python
import os
import argparse

# Numpy
import numpy as np
import matplotlib.pyplot as plt

# Opencv
import cv2

# lmdb
import lmdb

# Caffe
import caffe
from caffe.proto import caffe_pb2

# Project
from common.helper import is_lmdb_file, write_binproto_image, read_binproto_image


RESOURCES = os.path.abspath(os.path.join(os.path.dirname(__file__), 'resources'))
assert os.path.exists(RESOURCES), "Resources path is not found"


def compute_from_filelist(filelist, shape):
    mean_image = np.zeros(shape, dtype=np.float32)
    count = 0
    with open(filelist, 'r') as reader:
        for line in iter(lambda: reader.readline(), ''):
            f = line.split(' ')[0]
            assert os.path.exists(f), "Image path '%s' from file list is not found" % f
            img = cv2.imread(f)
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            rgb = cv2.resize(rgb, (shape[1], shape[2]))
            mean_image += rgb
            count += 1
    mean_image /= count
    return mean_image


def compute_from_lmdb(lmdb_file, shape):

    mean_image = np.zeros(shape, dtype=np.float32)
    count = 0
    env = lmdb.open(lmdb_file, readonly=True)
    with env.begin() as txn:
        datum = caffe.proto.caffe_pb2.Datum()
        cursor = txn.cursor()
        for key, value in cursor:
            datum.ParseFromString(value)
            rgb = caffe.io.datum_to_array(datum)
            # Reshape data (K, H, W) -> (H, W, K)
            rgb = rgb.T
            if rgb.shape[0] != shape[0] and rgb.shape[1] != shape[1]:
                rgb = cv2.resize(rgb, (shape[1], shape[0]))
            if args.verbose:
                img = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
                cv2.imshow("Current image", img)
                cv2.waitKey(150)
            mean_image += rgb
            count += 1
    mean_image /= count
    cv2.destroyAllWindows()
    env.close()
    return mean_image


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('file', type=str, help='Input image list txt file or lmdb file')
    parser.add_argument('--shape', type=int, nargs=3, default=(256, 256, 3), help='Output mean image shape (h, w, ch)')
    parser.add_argument('--verbose', action='store_true')

    args = parser.parse_args()

    assert os.path.exists(args.file), "Input file is not found"

    if is_lmdb_file(args.file):
        mean_image = compute_from_lmdb(args.file, args.shape)
    else:
        mean_image = compute_from_filelist(args.file, args.shape)

    if args.verbose:
        plt.title("Mean image to write")
        plt.imshow(mean_image.astype(np.uint8))
        plt.show()
        print mean_image.dtype, mean_image.shape, np.min(mean_image), np.max(mean_image), np.mean(mean_image)

    filename = os.path.join(RESOURCES, 'mean_image_' + os.path.basename(args.file) + '.binproto')
    if os.path.exists(filename):
        print "Remove existing mean image file %s" % filename
        os.remove(filename)
    write_binproto_image(mean_image, filename)

    # Test if data is correctly written:
    data = read_binproto_image(filename)
    data = data[0, :, :, :].T
    if args.verbose:
        print data.shape, data.dtype, mean_image.shape, mean_image.dtype
        print data.dtype, data.shape, np.min(data), np.max(data), np.mean(data)

    assert np.sum(mean_image - data) < 1e-7, "Read data is not equal to the written data"