
#
# Test a trained model 
#

# Python
import os


RESOURCES = os.path.abspath(os.path.join(os.path.dirname(__file__), 'resources'))
MODELS = os.path.abspath(os.path.join(os.path.dirname(__file__), 'models'))
PYCAFFE_PATH = "/Users/vfomin/Documents/ML/caffe-master_42cd785/python"

CAFFE_MODEL_FILE_PATH = os.path.join(MODELS, "deploy_caffenet-ddd_finetunning.prototxt")
CAFFE_WEIGHTS_FILE_PATH = os.path.join(RESOURCES, "train_val_caffenet-ddd_finetunning_iter_500.caffemodel")


assert os.path.exists(PYCAFFE_PATH), "PyCaffe path is not found"
assert os.path.isfile(CAFFE_MODEL_FILE_PATH), "Caffe model file is not found"
assert os.path.isfile(CAFFE_WEIGHTS_FILE_PATH), "Caffe model weights file is not found"

# Numpy
import numpy as np

# MPL
import matplotlib.pyplot as plt

# Sklearn
from sklearn.metrics import classification_report, confusion_matrix, log_loss

# Caffe
#sys.path.insert(0, PYCAFFE_PATH)
import caffe

# Project
from helper import read_binproto_image, lmdb_images_iterator


net = caffe.Net(CAFFE_MODEL_FILE_PATH, CAFFE_WEIGHTS_FILE_PATH, caffe.TEST)

# mean_image_path = os.path.join(RESOURCES, 'mean_image_train__2_2_10_75_0__224_224.lmdb.binproto')
mean_image_path = "/Users/vfomin/Documents/ML/caffe-master_42cd785/data/ilsvrc12/imagenet_mean.binaryproto"
mean_image = read_binproto_image(mean_image_path)[0, :, :, :]  # (K, H, W)
# crop to data size :
mean_image = mean_image[:, :227, :227]
# transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
# print transformer.inputs['data']
# transformer.set_mean('data', mean_image)
# net.blobs['data'].reshape(1, 3, 227, 227)

val_data_path = os.path.join(RESOURCES, "train__d0_12__c10_n10__s256_256.lmdb")
# val_data_path = os.path.join(RESOURCES, "train__d0_3__c10_n10__s256_256.lmdb")
# val_data_path = os.path.join(RESOURCES, "val__d13_25__c10_n10__s256_256.lmdb")
# val_data_path = os.path.join(RESOURCES, "test__100__256_256.lmdb")


def compute_stats():

    test_targets = []
    target_pred = []
    target_proba_pred = []

    for data, label in lmdb_images_iterator(val_data_path):

        data = data[:, :227, :227]
        transformed_image = data - mean_image
        net.blobs['data'].data[...] = transformed_image

        output = net.forward()
        output_prob = output['prob'] # the output probability vector for the first image in the batch

        test_targets.append(label)
        target_pred.append(output_prob.argmax())
        target_proba_pred.extend(output_prob)

    cr = classification_report(test_targets, target_pred)
    print cr
    cm = confusion_matrix(test_targets, target_pred, labels=range(10))
    print cm

    score = log_loss(test_targets, target_proba_pred)
    print 'Score log_loss: ', score


def show_classification():
    for data, label in lmdb_images_iterator(val_data_path):
        # data.shape = (K, H, W)
        data = data[:, :227, :227]
        im = data.T.astype(np.uint8).copy()
        # transformed_image = transformer.preprocess('data', data)
        transformed_image = data - mean_image

        net.blobs['data'].data[...] = transformed_image
        output = net.forward()
        output_prob = output['prob'] # the output probability vector for the first image in the batch
        output_acc = output['accuracy']  # the output probability vector for the first image in the batch
        print 'predicted class is:', output_prob.argmax(), output_prob
        print 'Accuracy : ', output_acc
        print 'Truth label : ', label

        # plt.subplot(121)
        plt.imshow(im)
        # plt.subplot(122)
        # plt.imshow(transformed_image.T)
        plt.show()


compute_stats()
# compute_stats()
# show_classification()





