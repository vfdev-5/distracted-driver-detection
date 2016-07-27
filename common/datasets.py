
# Python
import os
import random
from glob import glob

# Pandas
import pandas as pd

# Sklearn
from sklearn.cross_validation import train_test_split


INPUT_PATH = '../input'
DRIVER_IMGS_LIST_CSV = os.path.join(INPUT_PATH, 'driver_imgs_list.csv')
TRAIN_IMGS_PATH = os.path.abspath(os.path.join(INPUT_PATH, 'train'))

assert os.path.exists(DRIVER_IMGS_LIST_CSV), "Please configure the path to 'driver_imgs_list.csv' file"
assert os.path.exists(TRAIN_IMGS_PATH), "Please configure the path to train images folder"


def get_drivers_list():
    """
    :returns a list of available drivers
    """
    drivers_list_df = pd.read_csv(DRIVER_IMGS_LIST_CSV, ",")
    return drivers_list_df.subject.unique()


def trainval_files(classes, drivers, nb_samples, validation_size):
    """
    :param classes is a list of classes to select images, e.g (0, 1, 3)
    :param drivers is a list of drivers to select images, e.g. (p041, p026)
    :param nb_samples is a number of samples per class and per driver
    :param validation_size is a size of validation set over size of training set
    :returns training image files, training targets, validation image files, validation targets
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

    train_files = []
    train_targets = []
    validation_files = []
    validation_targets = []


    for driver in drivers:
        files, targets = _get_images_and_classes(driver, classes, nb_samples)
        t_files, v_files, t_targets, v_targets = train_test_split(files, targets, test_size=validation_size, random_state=42)
        train_files.extend(t_files)
        train_targets.extend(t_targets)
        validation_files.extend(v_files)
        validation_targets.extend(v_targets)

    return train_files, train_targets, validation_files, validation_targets


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

    classes = (0, 1, 2)
    nb_class_driver_samples = 3
    validation_size = 0.4

    sets = trainval_files(classes, all_drivers, nb_class_driver_samples, validation_size)
    print len(sets[0]), sets[0]
    print len(sets[1]), sets[1]
    print len(sets[2]), sets[2]
    print len(sets[3]), sets[3]
