#!/bin/python2

# Python
import os
import argparse

# Numpy
import numpy as np

# Opencv
import cv2

# lmdb
import lmdb

# Caffe
import caffe


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
            mean_image += rgb.T
            count += 1
    mean_image /= count
    return mean_image


def compute_from_lmdb(lmdb_file, shape):
    mean_image = np.zeros(shape)
    count = 0
    env = lmdb.open(lmdb_file, readonly=True)
    with env.begin() as txn:
        datum = caffe.proto.caffe_pb2.Datum()
        cursor = txn.cursor()
        for key, value in cursor:
            datum.ParseFromString(value)
            flat_img = np.fromstring(datum.data, dtype=np.uint8)
            rgb = flat_img.reshape((datum.height, datum.width, datum.channels))
            if rgb.shape[0] != shape[2] and \
                rgb.shape[1] != shape[1]:
                rgb = cv2.resize(rgb, (shape[1], shape[2]))
            mean_image += rgb.T
            count += 1
    mean_image /= count
    env.close()
    return mean_image


def write_binary_proto_file(mean_image, filename):
    mean_image = mean_image.reshape((1, ) + mean_image.shape)
    blob = caffe.io.array_to_blobproto(mean_image).SerializeToString()
    with open(filename, 'wb') as f:
        f.write(blob)


def is_lmdb_file(filename):
    try:
        env = lmdb.open(filename, readonly=True)
        env.close()
    except lmdb.Error:
        return False
    return True


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('file', type=str, help='Input image list txt file or lmdb file')
    parser.add_argument('--shape', type=int, nargs=3, default=(3, 256, 256), help='Output mean image shape (ch, w, h)')
    parser.add_argument('--verbose', action='store_true')

    args = parser.parse_args()

    assert os.path.exists(args.file), "Input file is not found"

    if is_lmdb_file(args.file):
        mean_image = compute_from_lmdb(args.file, args.shape)
    else:
        mean_image = compute_from_filelist(args.file, args.shape)

    if args.verbose:
        import matplotlib.pyplot as plt
        plt.imshow(mean_image.T.astype(np.uint8))
        plt.show()
        print mean_image.dtype, mean_image.shape, np.min(mean_image), np.max(mean_image), np.mean(mean_image)

    filename = os.path.join('resources', 'mean_image_' + os.path.basename(args.file) + '.binproto')
    write_binary_proto_file(mean_image, filename)
