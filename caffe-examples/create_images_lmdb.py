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


def create_trainval_lmdb(count, randomize, size, valsize):

    all_files = []
    all_labels = []
    for cls in xrange(10):
        files = glob(join(INPUT_DATA_PATH, 'train', 'c%i' % cls, '*.jpg'))
        if count > 0:
            nb_per_class = max(int(count / 10), 1)
            if randomize:
                files = random.sample(files, nb_per_class)
            else:
                files = files[0:nb_per_class]
        all_files.extend(files)
        all_labels.extend([cls]*len(files))

    train_files, val_files, train_labels, val_labels = train_test_split(
        all_files, all_labels, test_size=valsize, random_state=42
    )

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


def create_test_lmdb(count, randomize, size):

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
    parser.add_argument('type', type=str, choices=('trainval', 'test'),
                        help='Type of the file list')
    parser.add_argument('--count', type=int, default=-1,
                        help='Take \'count\' images. If \'count\' = -1 take all images')
    parser.add_argument('--randomize', action='store_true',
                        help="Choose randomly images if \'count\' > 0")
    parser.add_argument('--size', type=int, nargs=2, default=None,
                        help="Output image size : width height")
    parser.add_argument('--val-size', type=float, default=0.25,
                        help="Validation data percentage. Default : 0.25")

    args = parser.parse_args()
    print args

    if args.type == 'trainval':
        create_trainval_lmdb(args.count, args.randomize, args.size, args.val_size)
    elif args.type == 'test':
        create_test_lmdb(args.count, args.randomize, args.size)

    print "Output file is written in 'resources' folder"


