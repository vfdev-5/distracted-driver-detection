# Python
from os import path
from os.path import join
import shutil
import argparse
from glob import glob
import random

# Numpy
import numpy as np

# Matplotlib
import matplotlib.pyplot as plt

# Sklearn
from sklearn.cross_validation import train_test_split

# Opencv
import cv2

# Lmdb and Caffe
import lmdb
import caffe


# Project
from common.datasets import trainval_files


INPUT_DATA_PATH = path.abspath(path.join(path.dirname(__file__), "..", "input"))
assert path.exists(INPUT_DATA_PATH), "INPUT_DATA_PATH is not properly configured"


def read_image(filename, size):
    assert path.exists(filename), "Image path '%s' from file list is not found" % filename
    img = cv2.imread(filename)
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if size is not None:
        rgb = cv2.resize(rgb, (size[0], size[1]))
    return rgb


def create_lmdb(lmdb_filepath, input_files, labels, size):
    """
    Write images from input_files and labels in lmdb_filepath
    """
    if path.exists(lmdb_filepath):
        shutil.rmtree(lmdb_filepath)
    # maximum size of the whole DB as 10 times bigger then size of images
    estimate_map_size = (3 * size[0] * size[1]) * len(input_files) * 10
    env = lmdb.open(lmdb_filepath, estimate_map_size)
    with env.begin(write=True) as txn:
        counter = 0
        if labels is None:
            labels = [None]*len(input_files)
        for f, cls in zip(input_files, labels):
            # Read data from file :
            data = read_image(f, size)

            datum = caffe.proto.caffe_pb2.Datum()
            datum.channels = data.shape[2]
            datum.height = data.shape[0]
            datum.width = data.shape[1]
            datum.data = data.tobytes()

            if cls is not None:
                datum.label = int(cls)
            str_id = '{:08}'.format(counter)
            txn.put(str_id.encode('ascii'), datum.SerializeToString())
            counter += 1
    env.close()


def create_trainval_lmdb(args):


    sets = trainval_files(classes, drivers, args.nb_samples, args.validation_size, val_drivers)


    # Write train lmdb
    filename = 'train_%i' % count if count > 0 else 'train_all'
    if size is not None:
        filename += '__%i_%i' % (size[0], size[1])
    filename += '.lmdb'
    filepath = join('resources', filename)
    create_lmdb(filepath, train_files, train_labels, size)

    # Write val lmdb
    filename = 'val_%i' % count if count > 0 else 'val_all'
    if size is not None:
        filename += '__%i_%i' % (size[0], size[1])
    filename += '.lmdb'
    filepath = join('resources', filename)
    create_lmdb(filepath, val_files, val_labels, size)


def create_test_lmdb(args):

    files = glob(path.join(INPUT_DATA_PATH, 'test', '*.jpg'))
    if count > 0:
        if randomize:
            files = random.sample(files, count)
        else:
            files = files[0:count]

    # Write train lmdb
    filename = 'test_%i' % count if count > 0 else 'test_all'
    if size is not None:
        filename += '__%i_%i' % (size[0], size[1])
    filename += '.lmdb'
    filepath = join('resources', filename)
    create_lmdb(filepath, files, None, size)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--nb-samples', type=int, default=10,
                        help='Number of samples per class and per driver')
    parser.add_argument('--validation-size', type=float, default=0.4,
                        help="Validation size between 0 to 1")
    parser.add_argument('--nb-classes', type=int, default=10,
                        help="Number of classes between 1 to 10. All classes : -1")
    parser.add_argument('--nb-drivers', type=int, default=20,
                        help="Number of drivers in training/validation sets, between 1 to 26. All drivers : -1")
    parser.add_argument('--nb-val-drivers', type=int, default=6,
                        help="Number of drivers in validation only sets. nb-drivers + nb-val-drivers should less or equal than 26.")
    parser.add_argument('--size', type=int, nargs=2, default=None,
                        help="Output image size : width height")

    args = parser.parse_args()
    print args

    if args.type == 'trainval':
        create_trainval_lmdb(args)
    elif args.type == 'test':
        create_test_lmdb(args)

    print "Output file is written in 'resources' folder"


