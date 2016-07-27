
# Python
import os
import random
from itertools import groupby

# Pandas
import pandas as pd

# Sklearn
from sklearn.cross_validation import train_test_split


INPUT_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__),'../input'))

DRIVER_IMGS_LIST_CSV = os.path.join(INPUT_PATH, 'driver_imgs_list.csv')
TRAIN_IMGS_PATH = os.path.abspath(os.path.join(INPUT_PATH, 'train'))

assert os.path.exists(DRIVER_IMGS_LIST_CSV), "Please configure the path to 'driver_imgs_list.csv' file. Current path is %s" % DRIVER_IMGS_LIST_CSV
assert os.path.exists(TRAIN_IMGS_PATH), "Please configure the path to train images folder"


def get_drivers_list():
    """
    :returns a list of available drivers
    """
    drivers_list_df = pd.read_csv(DRIVER_IMGS_LIST_CSV, ",")
    return drivers_list_df.subject.unique()


def trainval_files(classes, drivers, nb_samples, validation_size, val_drivers=None, return_drivers=False):
    """
    :param classes is a list of classes to select images, e.g (0, 1, 3)
    :param drivers is a list of drivers to select images, e.g. (p041, p026)
    :param nb_samples is a number of samples per class and per driver
    :param validation_size is a size of validation set over size of training set
    :param val_drivers is a list of drivers only for the validation set. Values can be different from drivers. Can be None if same drivers for training and validation sets
    :param return_drivers is a flag to return two additional lists with drivers for training and validation sets
    :returns training image files, training targets, validation image files, validation targets, training drivers, validation drivers
    """

    drivers_list_df = pd.read_csv(DRIVER_IMGS_LIST_CSV, ",")

    def _get_images_and_classes(driver, classes, nb_samples):
        files = []
        targets = []
        imgs_classes_df = drivers_list_df[
            drivers_list_df.subject == driver
            ]
        for cls in classes:
            imgs_df = imgs_classes_df[imgs_classes_df.classname == 'c%i' % cls]
            imgs = imgs_df.img.unique()
            if nb_samples < len(imgs):
                imgs = random.sample(imgs, nb_samples)
            img_files = [os.path.join(TRAIN_IMGS_PATH, 'c%i' % cls, img) for img in imgs]
            files.extend(img_files)
            targets.extend([cls]*len(img_files))

        return files, targets

    def _get_files_targets_drivers(drivers):
        _files = []
        _targets = []
        _list_drivers = []

        for driver in drivers:
            f, t = _get_images_and_classes(driver, classes, nb_samples)
            _files.extend(f)
            _targets.extend(t)
            _list_drivers.extend([driver]*len(f))
        return _files, _targets, _list_drivers

    files, targets, list_drivers = _get_files_targets_drivers(drivers)

    train_files, validation_files, \
    train_targets, validation_targets, \
    train_drivers, validation_drivers = \
        train_test_split(files, targets, list_drivers, test_size=validation_size, random_state=42)

    if val_drivers is not None:
        files, targets, list_drivers = _get_files_targets_drivers(val_drivers)
        _, v_f, \
        _, v_t, \
        _, v_d = \
            train_test_split(files, targets, list_drivers, test_size=validation_size, random_state=42)
        validation_files.extend(v_f)
        validation_targets.extend(v_t)
        validation_drivers.extend(v_d)

    if not return_drivers:
        return train_files, train_targets, validation_files, validation_targets
    else:
        return train_files, train_targets, validation_files, validation_targets, train_drivers, validation_drivers


def write_images_list(files, targets, output_filename):
    """
    :param files list of pathes of images
    :param targets list of targets corresponding to files
    :param output_filename name of the output list file
    """
    assert len(files) == len(targets), "Length of input files and input targets should be equal"
    assert not os.path.exists(output_filename), "Output list file should not exist"

    with open(output_filename, 'w') as writer:
        for f, l in zip(files, targets):
            writer.write('%s %i\n' % (f, l))


def write_datasets_lists(sets, output_train_filename, output_test_filename):
    """
    :param sets a list of datasets and labels : [train_files, train_targets, validation_files, validation_targets]
    :param output_train_filename output list file name with pathes of train images
    :param output_test_filename output list file name with pathes of test images
    """
    assert isinstance(sets, tuple) and len(sets) == 4, "Input sets should be a list of length equal 4"
    assert len(sets[0]) == len(sets[1]), "Lenght of lists of train files (sets[0]) and train targets (sets[1]) should be equal"
    assert len(sets[2]) == len(sets[3]), "Lenght of lists of validation files (sets[2]) and validation targets (sets[3]) should be equal"
    assert not os.path.exists(output_train_filename), "Output file with train images should not exist"
    assert not os.path.exists(output_test_filename), "Output file with test images files should not exist"

    write_images_list(sets[0], sets[1], output_train_filename)
    write_images_list(sets[2], sets[3], output_test_filename)


if __name__ == "__main__":

    all_drivers = get_drivers_list()

    drivers = all_drivers[:4]
    #val_drivers = all_drivers[5:7]
    val_drivers = []
    classes = (0, 1, 2,)
    nb_class_driver_samples = 10
    validation_size = 0.4

    print "total number of images : ", len(drivers) * len(classes) * nb_class_driver_samples + \
                            int(len(val_drivers) * len(classes) * nb_class_driver_samples * validation_size + 0.5)
    print drivers, val_drivers

    sets = trainval_files(classes, drivers, nb_class_driver_samples, validation_size, val_drivers, True)
    print "Train images : ", len(sets[0]), sets[0]
    print "Train labels : ", len(sets[1]), sets[1]
    print "Train drivers : ", len(sets[4]), sets[4]
    print "Test images : ", len(sets[2]), sets[2]
    print "Test labels : ", len(sets[3]), sets[3]
    print "Test drivers : ", len(sets[5]), sets[5]

    print "Train label frequencies : "
    sets[1].sort()
    for key, group in groupby(sets[1]):
        print "label", key, ":", len(list(group))

    print "Test label frequencies : "
    sets[3].sort()
    for key, group in groupby(sets[3]):
        print "label", key, ":", len(list(group))
