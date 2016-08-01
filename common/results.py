#
# Script to predict results using a trained classifier
#

# Python
import logging
import os
from glob import glob
from time import time

# Numpy
import numpy as np

# Project
from common.preprocessing import get_data_parallel2, get_data2

INPUT_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '../input'))
CLASSES = range(10)

logger = logging.getLogger()
logger.setLevel(logging.INFO)


def _write_predictions_header(filename, header_list):
    header_str = ",".join(header_list) + '\n'
    with open(filename, 'w') as writer:
        writer.write(header_str)


def _write_predictions_block(filename, data_array):
    with open(filename, 'a') as writer:
        for row in data_array:
            line = ",".join([str(item) for item in row]) + '\n'
            writer.write(line)


def predict_and_write(width, height, prediction_func, output_filename):

    if os.path.exists(output_filename):
        logging.error("Submission file is already existing")
        return

    ###########################################################################
    logging.info("- Predict and write submission file")
    ###########################################################################

    start = time()
    files = np.array(glob(os.path.join(INPUT_PATH, "test", "*.jpg")))
    # Process files by blocks of 500 files
    block_size = 500
    index = 0

    header = ['img', ]
    header.extend(['c'+str(cls) for cls in CLASSES])

    _write_predictions_header(output_filename, header)

    logging.info("-- Total number of files : {}".format(len(files)))

    while index < len(files):
        if index+block_size >= len(files):
            block_size = len(files) - index
        block_files = files[index:index+block_size]
        index += block_size

        logging.info("-- setup test data. block index = {}".format(index))

        X_test = get_data2(block_files, width, height)

        logging.info("-- predict classes")

        target_proba_pred = prediction_func(X_test)
        # assert isinstance(target_proba_pred, np.ndarray), "target_proba_pred should be a ndarray"

        data_array = target_proba_pred.tolist()
        for l, f in zip(data_array, block_files):
            l.insert(0, os.path.basename(f))
        _write_predictions_block(output_filename, data_array)
        raise Exception()

    logging.info("Elapsed seconds : {}".format(time() - start))

