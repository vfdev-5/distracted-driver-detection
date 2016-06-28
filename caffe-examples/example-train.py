#!/bin/python2

# Python
import os

# Caffe
import caffe

# ***************** Configuration ********************
caffe.set_mode_cpu()
NET_FILE = os.path.abspath(os.path.join('resources', 'net.prototxt'))

# ****************************************************
assert os.path.exists(NET_FILE), "Net definition file is not found"

net = caffe.Net(NET_FILE, caffe.TRAIN)

print net.layers