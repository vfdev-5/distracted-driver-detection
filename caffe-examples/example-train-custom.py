#!/bin/python2

#
# Simple logistic regression example using Caffe
#

# Python
import os

# Numpy
import numpy as np


RESOURCES = '/home/osboxes/Documents/state-farm-distracted-driver-detection/caffe-examples/resources'

assert os.path.exists(os.path.join(RESOURCES, 'train_list_100.txt')), "Train list file is not found"
assert os.path.exists(os.path.join(RESOURCES, 'train_list_50.txt')), "Test list file is not found"

#
### Setup Caffe
#
import os
import caffe


solver = caffe.get_solver(os.path.join(RESOURCES, 'solver.prototxt'))


niter = 250
test_interval = niter / 10
# losses will also be stored in the log
train_loss = np.zeros(niter)
test_acc = np.zeros(int(np.ceil(niter / test_interval)))

# the main solver loop
for it in range(niter):
    solver.step(1)  # SGD by Caffe

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


# print solver.net.forward()  # train net
# Train the net
# solver.solve()
#
# print solver.net.forward()
# print solver.test_nets[0].forward()
