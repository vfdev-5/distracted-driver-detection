
#
# Helper methods
#


# Python
import sys
import os

# Numpy & Matplotlib
import numpy as np
import matplotlib.pyplot as plt

# Caffe & lmdb
import caffe
import lmdb
from caffe.proto import caffe_pb2


def is_lmdb_file(filename):
    try:
        env = lmdb.open(filename, readonly=True)
        env.close()
    except lmdb.Error:
        return False
    return True


def write_binproto_image(data, filename):
    """
    :param data: image data with shape, e.g. (height, width, channel)
    :param filename: output filename
    """
    data = data.T
    data = data.reshape((1, ) + data.shape)
    blob = caffe.io.array_to_blobproto(data).SerializeToString()
    with open(filename, 'wb') as f:
        f.write(blob)


def read_binproto_image(filename):
    """
    :param filename: input binproto filename
    :return: ndarray with shape (Batch size, Height, Width, Channels)
    """
    blob = caffe_pb2.BlobProto()
    with open(filename, 'rb') as f:
        blob.ParseFromString(f.read())
    return caffe.io.blobproto_to_array(blob)


def display_binproto_images(binproto_file):
    data = read_binproto_image(binproto_file)
    print data.shape, data.dtype
    assert len(data.shape) == 4, "Provided data in binproto file is not recognized as a Caffe Blob"

    for i in range(data.shape[0]):
        data = data[i, :, :, :].T.astype(np.uint8)
        plt.title("Binproto image")
        plt.imshow(data)
        plt.show()


def lmdb_images_iterator(lmdb_file):
    env = lmdb.open(lmdb_file)
    with env.begin() as lmdb_txn:
        lmdb_cursor = lmdb_txn.cursor()
        datum = caffe_pb2.Datum()
        for key, value in lmdb_cursor:
            datum.ParseFromString(value)
            label = datum.label
            data = caffe.io.datum_to_array(datum)
            yield data, label
    env.close()


def display_lmdb_images(lmdb_file):
    counter = 0
    for data, label in lmdb_images_iterator(lmdb_file):
        im = data.T.astype(np.uint8)
        print "label ", label
        plt.title("Lmdb image : %i/N" % counter)
        plt.imshow(im)
        plt.show()


if __name__ == "__main__":

    assert len(sys.argv) > 1, "Please provide a file as argument"
    filepath = sys.argv[1]
    assert os.path.exists(filepath), "Provided file is not found"

    if is_lmdb_file(filepath):
        display_lmdb_images(filepath)
    elif filepath.split('.')[-1] == 'binproto':
        display_binproto_images(filepath)
    else:
        raise Exception("Provided file type is not recognized")

