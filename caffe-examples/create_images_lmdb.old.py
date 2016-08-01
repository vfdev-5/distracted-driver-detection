# Python
import os
import argparse

# Project
from common.datasets import trainval_files, test_files, get_drivers_list, write_images_lmdb


RESOURCES = os.path.abspath(os.path.join(os.path.dirname(__file__), 'resources'))
assert os.path.exists(RESOURCES), "Resources path is not found"


def create_trainval_lmdb(args):

    classes = range(0, args.nb_classes if 0 < args.nb_classes < 11 else 10)
    all_drivers = get_drivers_list()
    assert args.nb_drivers + args.nb_val_drivers <= len(all_drivers) and \
           args.nb_drivers > 0 and args.nb_val_drivers >= 0, \
        "nb-drivers + nb-val-drivers should or equal less than %i" % len(all_drivers)

    drivers = all_drivers[:args.nb_drivers if 0 < args.nb_drivers <= len(all_drivers) else len(all_drivers)]

    if args.nb_val_drivers > 0:
        val_drivers = all_drivers[args.nb_drivers:args.nb_drivers+args.nb_val_drivers]
    else:
        val_drivers = []

    nb_train = int(len(classes) * len(drivers) * args.nb_samples * (1.0 - args.validation_size) + 0.5)
    nb_val = int(len(classes) * (len(drivers) + len(val_drivers) if val_drivers is not None else 0) * args.nb_samples * args.validation_size + 0.5)
    print "Estimate number of train images : ", nb_train
    print "Estimate number of validation images : ", nb_val

    sets = trainval_files(classes, drivers, args.nb_samples, args.validation_size, val_drivers)

    output_train_filename = os.path.join(RESOURCES, 'train__%i_%i_%i_%i_%i'
                                         % (args.nb_drivers,
                                            args.nb_classes,
                                            args.nb_samples,
                                            int(100*(1.0-args.validation_size)),
                                            len(val_drivers)))
    if args.size is not None:
        output_train_filename += '__%i_%i' % (args.size[0], args.size[1])
    output_train_filename += '.lmdb'

    output_val_filename = os.path.join(RESOURCES, 'val__%i_%i_%i_%i_%i'
                                       % (args.nb_drivers,
                                          args.nb_classes,
                                          args.nb_samples,
                                          int(100*args.validation_size),
                                          len(val_drivers)))
    if args.size is not None:
        output_val_filename += '__%i_%i' % (args.size[0], args.size[1])
    output_val_filename += '.lmdb'

    # Write train lmdb
    write_images_lmdb(output_train_filename, sets[0], sets[1], args.size)

    # Write val lmdb
    write_images_lmdb(output_val_filename, sets[2], sets[3], args.size)


def create_test_lmdb(args):

    files = test_files(args.nb_samples)

    # Write train lmdb
    output_filename = os.path.join(RESOURCES, 'test__%i'
                                              % (args.nb_samples))
    if args.size is not None:
        output_filename += '__%i_%i' % (args.size[0], args.size[1])
    output_filename += '.lmdb'
    write_images_lmdb(output_filename, files, None, args.size)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('type', type=str, choices=('trainval', 'test'),
                        help='Type of the file list')
    parser.add_argument('--nb-samples', type=int, default=10,
                        help='Number of samples per class and per driver (type=trainval). Number of files to fetch (type=test)')
    parser.add_argument('--validation-size', type=float, default=0.25,
                        help="Validation data (on the same drivers) size between 0 to 1 (type=trainval)")
    parser.add_argument('--nb-classes', type=int, default=10,
                        help="Number of classes between 1 to 10 (type=trainval). All classes : -1")
    parser.add_argument('--nb-drivers', type=int, default=20,
                        help="Number of drivers in training/validation sets, between 1 to 26 (type=trainval). All drivers : -1")
    parser.add_argument('--nb-val-drivers', type=int, default=6,
                        help="Number of drivers in validation only sets (type=trainval). nb-drivers + nb-val-drivers should less or equal than 26.")
    parser.add_argument('--size', type=int, nargs=2, default=None,
                        help="Output image size : width height")

    args = parser.parse_args()
    print args

    if args.type == 'trainval':
        create_trainval_lmdb(args)
    elif args.type == 'test':
        create_test_lmdb(args)
    else:
        raise Exception("Type '%s' is not recognized" % args.type)

    print "Output file is written in 'resources' folder"


