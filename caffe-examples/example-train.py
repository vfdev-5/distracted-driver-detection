#!/bin/python2

#
# Simple logistic regression example using Caffe
#

# Python
import os

# Matplotlib
import matplotlib.pyplot as plt


RESOURCES = '/home/osboxes/Documents/state-farm-distracted-driver-detection/caffe-examples/resources'

assert os.path.exists(os.path.join(RESOURCES, 'train_100__10_10.lmdb')), "Train lmdb is not found"
assert os.path.exists(os.path.join(RESOURCES, 'val_100__10_10.lmdb')), "Validation lmdb is not found"

#
### Setup Caffe
#
import os
import caffe


solver = caffe.get_solver(os.path.join(RESOURCES, 'solver_small-conf.prototxt'))

print solver.net.forward()  # train net


# Train the net
solver.solve()

print solver.net.forward()
print solver.test_nets[0].forward()
