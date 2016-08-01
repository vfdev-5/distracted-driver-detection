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
from common.preprocessing import get_data_parallel2, get_raw_data

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


def predict_and_write(width, height, output_filename, prediction_func, file_reader_func=None):
    """
    Method to make predictions on test data and write submission file
    :param width: desirable image width of the test data
    :param height: desirable image height of the test data 
    :param prediction_func: function that is called on data_batches to make predictions.
            Function receives one argument: data_batch (ndarray) with a shape (number of files in batch, number of channels, height, width)
            should return

    :param file_reader_func: function that is called to open (and resize) a batch of files. If None a default function is used. Function receives 3 args: list of files, width, height
    The function should return an instance of ndarray with a shape (number of files in batch, number of channels, height, width)
    """
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

    _get_data = file_reader_func if file_reader_func is not None else get_raw_data

    while index < len(files):
        if index+block_size >= len(files):
            block_size = len(files) - index
        block_files = files[index:index+block_size]
        index += block_size

        logging.info("-- setup test data. block index = {}".format(index))
        
        X_test = _get_data(block_files, width, height)
        assert isinstance(X_test, np.ndarray) and len(X_test.shape) == 4, "Bad output from the function '_get_data'"

        logging.info("-- predict classes")

        target_proba_pred = prediction_func(X_test)
        assert isinstance(target_proba_pred, np.ndarray) and \
               target_proba_pred.shape[0] == X_test.shape[0], \
            "target_proba_pred should be a ndarray with the shape (block_size, nb_classes)"

        data_array = target_proba_pred.tolist()
        for l, f in zip(data_array, block_files):
            l.insert(0, os.path.basename(f))
        _write_predictions_block(output_filename, data_array)

    logging.info("Elapsed seconds : {}".format(time() - start))

