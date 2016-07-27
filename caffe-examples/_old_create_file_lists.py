#!/bin/python2

from os import path
from glob import glob
import random
import argparse

INPUT_DATA_PATH = path.abspath(path.join(path.dirname(__file__), "..", "input"))
assert path.exists(INPUT_DATA_PATH), "INPUT_DATA_PATH is not properly configured"


def create_train_list(count, randomize):
    filename = 'train_list_%i.txt' % count if count > 0 else 'train_list_all.txt'
    with open(path.join('resources', filename), 'w') as writer:
        for cls in xrange(10):
            files = glob(path.join(INPUT_DATA_PATH, 'train', 'c%i' % cls, '*.jpg'))
            if count > 0:
                nb_per_class = max(int(count / 10), 1)
                if randomize:
                    files = random.sample(files, nb_per_class)
                else:
                    files = files[0:nb_per_class]
            for f in files:
                writer.write('%s %i\n' % (f, cls))


def create_test_list(count, randomize):
    filename = 'test_list_%i.txt' % count if count > 0 else 'test_list_all.txt'
    with open(path.join('resources', filename), 'w') as writer:
        files = glob(path.join(INPUT_DATA_PATH, 'test', '*.jpg'))
        if count > 0:
            if randomize:
                files = random.sample(files, count)
            else:
                files = files[0:count]
        for f in files:
            writer.write('%s\n' % f)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('type', type=str, choices=('train', 'test'),
                        help='Type of the file list')
    parser.add_argument('--count', type=int, default=-1,
                        help='Take \'count\' images. If \'count\' = -1 take all images')
    parser.add_argument('--randomize', action='store_true',
                        help="Choose randomly images if \'count\' > 0")

    args = parser.parse_args()
    print args

    if args.type == 'train':
        create_train_list(args.count, args.randomize)
    elif args.type == 'test':
        create_test_list(args.count, args.randomize)

    print "Output file is written in 'resources' folder"


