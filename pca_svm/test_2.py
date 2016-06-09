#
# Script to predict results using trained SVM Classifier
#

# Python
import os
import logging
from time import time
from datetime import datetime

# Numpy
import numpy as np

# Sklearn
from sklearn.externals import joblib
from sklearn.decomposition import RandomizedPCA

# Project
from common import get_data_parallel, get_data

logger = logging.getLogger()
logger.setLevel(logging.INFO)

###########################################################################
# DEFINE TRAINED MODEL
###########################################################################
classes = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9)
# number of images per class
n_samples_per_class = 150

# Image resize factor
resize_factor = 8

# Number of components for PCA
n_components = 100

###########################################################################
# Check for existing result
model_filename = 'svc_' + str(len(classes)) + '_' \
                 + str(len(classes) * n_samples_per_class) + '_' \
                 + str(n_components) + '_' \
                 + str(resize_factor) \
                 + '.pkl'

output_filename = 'submission_' + \
                  str(datetime.now().strftime("%Y-%m-%d-%H-%M")) + '__' + \
                  str(len(classes)) + '_' + \
                  str(len(classes) * n_samples_per_class) + '_' + \
                  str(n_components) + '_' + \
                  str(resize_factor) + \
                  '.csv'


if not os.path.exists(os.path.join('models', model_filename)):
    logging.error("Model is not found")
    exit(1)

if os.path.exists(os.path.join('results', output_filename)):
    logging.error("Submission file with given parameters is found")
    exit(1)


###########################################################################
logging.info("- Load trained model")
###########################################################################

start = time()

clf = joblib.load(os.path.join('models', model_filename))
pca = RandomizedPCA(n_components=n_components, whiten=True)

logging.info("Elapsed seconds : {}".format(time() - start))

print("Best estimator found by grid search:")
print(clf.best_estimator_)


###########################################################################
logging.info("- Predict and write submission file")
###########################################################################


start = time()
files = []
path = "../input/test"
for f in os.listdir(path):
    files.append(os.path.join(path, f))

files = np.array(files)
# Process files by blocks of 500 files
block_size = 500
index = 0


def write_predictions_header(filename, header_list):
    header_str = ",".join(header_list) + '\n'
    with open(filename, 'w') as writer:
        writer.write(header_str)


def write_predictions_block(filename, data_array):
    with open(filename, 'a') as writer:
        for row in data_array:
            line = ",".join([str(item) for item in row]) + '\n'
            writer.write(line)

header = ['img', ]
header.extend(['c'+str(cls) for cls in classes])

output_filename = os.path.join('results', output_filename)
write_predictions_header(output_filename, header)

n_classes = len(classes)

logging.info("-- Total number of files : {}".format(len(files)))

while index < len(files):
    if index+block_size >= len(files):
        block_size = len(files) - index
    block_files = files[index:index+block_size]
    index += block_size

    logging.info("-- setup test data. block index = {}".format(index))

    # t0 = time()
    # X_test_, width_, height_ = get_data(block_files, resize_factor)
    # print "get_data : ", time() - t0
    # t0 = time()
    X_test, width, height = get_data_parallel(block_files, resize_factor)
    # print "get_data_parallel : ", time() - t0
    # assert width == width_ and height == height_ and (X_test == X_test_).all(), "Data is not identical"

    logging.info("-- project the data on the eigenposes orthogonal basis")
    X_test_pca = pca.fit_transform(X_test)
    logging.info("-- predict classes")
    target_proba_pred = clf.predict_proba(X_test_pca)

    data_array = target_proba_pred.tolist()
    for l, f in zip(data_array, block_files):
        l.insert(0, os.path.basename(f))
    write_predictions_block(output_filename, data_array)


logging.info("Elapsed seconds : {}".format(time() - start))

