#!/bin/python2

import os
import argparse

# Project
from common.datasets import trainval_files, get_drivers_list, write_datasets_lists


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--nb-samples', type=int, default=10,
                        help='Number of samples per class and per driver')
    parser.add_argument('--validation-size', type=float, default=0.4,
                        help="Validation size between 0 to 1")

    args = parser.parse_args()
    print args

    classes = range(0, 10)
    drivers = get_drivers_list()

    nb_train = int(len(classes) * len(drivers) * args.nb_samples * (1.0 - args.validation_size))
    nb_test = int(len(classes) * len(drivers) * args.nb_samples * args.validation_size)

    output_train_list_filename = os.path.join('resources', 'train_list_%i.txt' % nb_train)
    output_test_list_filename = os.path.join('resources', 'test_list_%i.txt' % nb_test)

    sets = trainval_files(classes, drivers, args.nb_samples, args.validation_size)
    write_datasets_lists(sets, output_train_list_filename, output_test_list_filename)

    print "Output file is written in 'resources' folder"


