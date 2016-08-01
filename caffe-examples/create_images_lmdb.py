# Python
import os
import argparse

# Project
from common.datasets import trainval_files2, test_files, get_drivers_list, write_images_lmdb


RESOURCES = os.path.abspath(os.path.join(os.path.dirname(__file__), 'resources'))
assert os.path.exists(RESOURCES), "Resources path is not found"


def create_trainval_lmdb(args):

    classes = range(0, args.nb_classes if 0 < args.nb_classes < 11 else 10)
    all_drivers = get_drivers_list()
    ll = len(all_drivers)

    assert 0 <= args.train_drivers[0] < args.train_drivers[1]+1 <= ll, \
        "Train drivers indices should be between 0 and %i" % (ll-1)

    assert 0 <= args.val_drivers[0] < args.val_drivers[1]+1 <= ll, \
        "Validation drivers indices should be between 0 and %i" % (ll-1)

    train_drivers = all_drivers[args.train_drivers[0]:args.train_drivers[1]+1]

    if args.val_drivers[0] < args.val_drivers[1]:
        val_drivers = all_drivers[args.val_drivers[0]:args.val_drivers[1]+1]
    else:
        val_drivers = []

    nb_train = int(len(classes) * len(train_drivers) * args.nb_samples)
    nb_val = int(len(classes) * len(val_drivers) * args.nb_samples)
    print "Estimate number of train images : ", nb_train
    print "Estimate number of validation images : ", nb_val

    sets = trainval_files2(classes, train_drivers, val_drivers, args.nb_samples)

    output_train_filename = os.path.join(RESOURCES, 'train__d%i_%i__c%i_n%i'
                                         % (args.train_drivers[0],
                                            args.train_drivers[1],
                                            args.nb_classes,
                                            args.nb_samples))
    if args.size is not None:
        output_train_filename += '__s%i_%i' % (args.size[0], args.size[1])
    output_train_filename += '.lmdb'

    output_val_filename = os.path.join(RESOURCES, 'val__d%i_%i__c%i_n%i'
                                       % (args.val_drivers[0],
                                          args.val_drivers[1],
                                          args.nb_classes,
                                          args.nb_samples))
    if args.size is not None:
        output_val_filename += '__s%i_%i' % (args.size[0], args.size[1])
    output_val_filename += '.lmdb'

    # Write train lmdb
    write_images_lmdb(output_train_filename, sets[0], sets[1], args.size)

    # Write val lmdb
    write_images_lmdb(output_val_filename, sets[2], sets[3], args.size)


def create_test_lmdb(args):

    files = test_files(args.nb_samples)

    # Write train lmdb
    output_filename = 'test__%i' % (args.nb_samples) if args.nb_samples > 0 else 'test__all'
    output_filename = os.path.join(RESOURCES, output_filename)
    if args.size is not None:
        output_filename += '__%i_%i' % (args.size[0], args.size[1])
    output_filename += '.lmdb'
    write_images_lmdb(output_filename, files, None, args.size)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('type', type=str, choices=('trainval', 'test'),
                        help='Type of the file list')
    parser.add_argument('--train-drivers', type=int, nargs=2, default=[0, 12],
                        help="Drivers for the training set only. Two indices between 0 to 25 (type=trainval).")
    parser.add_argument('--val-drivers', type=int, nargs=2, default=[13, 25],
                    help="Drivers for the validation set only. Two indices between 0 to 25 (type=trainval).")
    parser.add_argument('--nb-samples', type=int, default=10,
                        help='Number of samples per class and per driver (type=trainval). Number of files to fetch (type=test)')
    parser.add_argument('--nb-classes', type=int, default=10,
                        help="Number of classes between 1 to 10 (type=trainval). All classes : -1")
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


