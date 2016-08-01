
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

    print data_batch.shape

    data = data_batch.reshape(
        data_batch.shape[0],
        227,
        227,
        3
    )
    target_proba_pred = []
    for img in data:

        transformed_image = img.T - mean_image
        net.blobs['data'].data[...] = transformed_image
        output = net.forward()
        output_prob = output['prob'] # the output probability vector for the first image in the batch

        target_proba_pred.extend(output_prob)

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

    predict_and_write(227, 227, predict, output_filename)





