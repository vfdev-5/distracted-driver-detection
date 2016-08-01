
#
#
#


# Python
import os
import sys
from datetime import datetime

RESOURCES = os.path.abspath(os.path.join(os.path.dirname(__file__), 'resources'))
MODELS = os.path.abspath(os.path.join(os.path.dirname(__file__), 'models'))
RESULTS = os.path.abspath(os.path.join(os.path.dirname(__file__), 'results'))
PYCAFFE_PATH = "/Users/vfomin/Documents/ML/caffe-master_42cd785/python"

CAFFE_MODEL_FILE_PATH = os.path.join(MODELS, "deploy_caffenet-ddd_finetunning.prototxt")
CAFFE_WEIGHTS_FILE_PATH = os.path.join(RESOURCES, "train_val_caffenet-ddd_finetunning_iter_500.caffemodel")


size = [227, 227]
classes = range(10)

assert os.path.exists(PYCAFFE_PATH), "PyCaffe path is not found"
assert os.path.isfile(CAFFE_MODEL_FILE_PATH), "Caffe model file is not found"
assert os.path.isfile(CAFFE_WEIGHTS_FILE_PATH), "Caffe model weights file is not found"

# Numpy
import numpy as np

# Caffe
sys.path.insert(0, PYCAFFE_PATH)
import caffe

# Project
from common.helper import read_binproto_image, lmdb_images_iterator
from common.results import predict_and_write


def predict(data_batch):
    """
    :param data_batch: shape=(number of files in batch, number of channels, height, width)
    :return:
    """
    net.blobs['data'].reshape(*data_batch.shape)

    mean_image_batch = mean_image[None, ...]
    mean_image_batch = np.repeat(mean_image_batch, data_batch.shape[0], axis=0)
    data_batch = data_batch - mean_image_batch
    net.blobs['data'].data[...] = data_batch

    output = net.forward()
    target_proba_pred = np.array(output['prob']) # the output probability vector for the first image in the batch
    return target_proba_pred


if __name__ == "__main__":

    net = caffe.Net(CAFFE_MODEL_FILE_PATH, CAFFE_WEIGHTS_FILE_PATH, caffe.TEST)

    mean_image_path = "/Users/vfomin/Documents/ML/caffe-master_42cd785/data/ilsvrc12/imagenet_mean.binaryproto"
    mean_image = read_binproto_image(mean_image_path)[0, :, :, :]  # (K, H, W)
    # crop to data size :
    mean_image = mean_image[:, :227, :227]

    output_filename = 'submission_' + \
                      str(datetime.now().strftime("%Y-%m-%d-%H-%M")) + '_' + \
                      os.path.basename(CAFFE_MODEL_FILE_PATH) + \
                      '.csv'

    output_filename = os.path.join(RESULTS, output_filename)

    predict_and_write(227, 227, output_filename, predict)





