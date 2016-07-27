#!/bin/python2

#
# Visualize model training
#

# Python
import os

# Numpy & Matplotlib
import numpy as np
import matplotlib.pyplot as plt

# Opencv
import cv2


RESOURCES = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'resources')
MODELS = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models')
SOLVER_PATH = os.path.join(MODELS, 'train_solver.prototxt')

assert os.path.exists(RESOURCES) and os.path.exists(MODELS) and os.path.exists(SOLVER_PATH), "Bad configuration"

#
### Setup Caffe
#
import caffe


#
# Check mean image
#
def check_mean_image():
    filename = os.path.join(RESOURCES, 'mean_image_train_list_72.txt.binproto')
    data = open(filename, 'rb').read()
    blob = caffe.proto.caffe_pb2.BlobProto()
    blob.ParseFromString(data)
    arr = np.array(caffe.io.blobproto_to_array(blob))
    img = arr[0, :, :, :]
    img = img.T.astype(np.uint8)
    plt.title('Mean img')
    plt.imshow(img)
    plt.show()

# check_mean_image()


caffe.set_mode_cpu()
solver = caffe.get_solver(SOLVER_PATH)

niter = 250
test_interval = niter / 10
# losses will also be stored in the log
train_loss = np.zeros(niter)
test_acc = np.zeros(int(np.ceil(niter / test_interval)))


# the main solver loop
for it in range(niter):

    print "-- Iteration", it

    solver.step(1)

    # Show input data :
    print solver.net.blobs['data'].data.shape

    for i in range(solver.net.blobs['data'].data.shape[0]):
        img = solver.net.blobs['data'].data[i, :, :, :].T.astype(np.uint8)
        print img.dtype, img.shape, img.min(), img.max()
        plt.title('data img %i' % i)
        plt.imshow(img)
        plt.show()



    break

    # store the train loss
    train_loss[it] = solver.net.blobs['loss'].data

    # run a full test every so often
    # (Caffe can also do this for us and write to a log, but we show here
    #  how to do it directly in Python, where more complicated things are easier.)
    if it % test_interval == 0:
        print 'Iteration', it, 'testing...'
        correct = 0
        for test_it in range(100):
            solver.test_nets[0].forward()
            correct += sum(solver.test_nets[0].blobs['score'].data.argmax(1) ==
                           solver.test_nets[0].blobs['label'].data)
        test_acc[it // test_interval] = correct / 1e4

    break


# print solver.net.forward()  # train net
# Train the net
# solver.solve()
#
# print solver.net.forward()
# print solver.test_nets[0].forward()
