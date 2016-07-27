#!/bin/python2

import os
import argparse

# Project
from common.datasets import trainval_files, get_drivers_list, write_datasets_lists

RESOURCES = os.path.abspath(os.path.join(os.path.dirname(__file__), 'resources'))
assert os.path.exists(RESOURCES), "Resources path is not found"

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--nb-samples', type=int, default=10,
                        help='Number of samples per class and per driver')
    parser.add_argument('--validation-size', type=float, default=0.4,
                        help="Validation size between 0 to 1")
    parser.add_argument('--nb-classes', type=int, default=10,
                        help="Number of classes between 1 to 10. All classes : -1")
    parser.add_argument('--nb-drivers', type=int, default=20,
                        help="Number of drivers in training/validation sets, between 1 to 26. All drivers : -1")
    parser.add_argument('--nb-val-drivers', type=int, default=6,
                        help="Number of drivers in validation only sets. nb-drivers + nb-val-drivers should less or equal than 26.")


    args = parser.parse_args()
    print args

    classes = range(0, args.nb_classes if args.nb_classes > 0 else 10)
    all_drivers = get_drivers_list()

    assert args.nb_drivers + args.nb_val_drivers <= len(all_drivers), "nb-drivers + nb-val-drivers should or equal less than %i" % len(all_drivers)

    drivers = all_drivers[:args.nb_drivers if args.nb_drivers > 0 else len(all_drivers)]
    if args.nb_val_drivers > 0:
        val_drivers = all_drivers[args.nb_drivers:args.nb_drivers+args.nb_val_drivers]
    else:
        val_drivers = []

    nb_train = int(len(classes) * len(drivers) * args.nb_samples * (1.0 - args.validation_size) + 0.5)
    nb_test = int(len(classes) * (len(drivers) + len(val_drivers) if val_drivers is not None else 0) * args.nb_samples * args.validation_size + 0.5)

    output_train_list_filename = os.path.join(RESOURCES, 'train_list_%i.txt' % nb_train)
    output_test_list_filename = os.path.join(RESOURCES, 'test_list_%i.txt' % nb_test)

    sets = trainval_files(classes, drivers, args.nb_samples, args.validation_size, val_drivers)
    write_datasets_lists(sets, output_train_list_filename, output_test_list_filename)

    print "Output file is written in 'resources' folder"


