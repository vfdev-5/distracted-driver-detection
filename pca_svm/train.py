#
# Script to train SVM Classifier on PCA reduced data
#

# Python
import os
import logging
import random
from time import time
from glob import glob

# Sklearn
from sklearn.cross_validation import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import log_loss
from sklearn.grid_search import GridSearchCV
from sklearn.decomposition import RandomizedPCA
from sklearn.svm import SVC
from sklearn.externals import joblib

# Project
from common import get_data_parallel, get_data


logger = logging.getLogger()
logger.setLevel(logging.INFO)

###########################################################################
# DEFINE GLOBAL TRAINING PARAMETERS
###########################################################################

classes = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9)
# number of images per class
n_samples_per_class = 175

# Image resize factor
resize_factor = 8

# Number of components for PCA
n_components = 150


###########################################################################
# Check for existing result
###########################################################################

model_filename = 'svc_' + str(len(classes)) + '_' \
                 + str(len(classes) * n_samples_per_class) + '_' \
                 + str(n_components) + '_' \
                 + str(resize_factor) \
                 + '.pkl'

pca_filename = 'pca_' + str(len(classes)) + '_' \
                 + str(len(classes) * n_samples_per_class) + '_' \
                 + str(n_components) + '_' \
                 + str(resize_factor) \
                 + '.pkl'

if os.path.exists(os.path.join('models', model_filename)):
    logging.error("Model is already trained for these parameters")
    exit(1)

if os.path.exists(os.path.join('models', pca_filename)):
    logging.error("PCA model is already trained for these parameters")
    exit(1)


###########################################################################
logging.info("- Prepare training data")
###########################################################################

targets = ()
files = []

for cls in classes:
    path = os.path.join("../input/train", 'c' + str(cls), "*.jpg")
    # choose randomly 'n_samples_per_class' files
    fls = random.sample(glob(path), n_samples_per_class)
    targets += (cls,) * len(fls)
    files.extend(fls)

n_classes = len(classes)
logging.info("data : classes={}, samples per class={}".format(classes, n_samples_per_class))


###########################################################################
logging.info("- Split into a training set and a test set")
###########################################################################

files_train, files_test, target_train, target_test = train_test_split(
    files, targets, test_size=0.25, random_state=42
)

logging.info("Train files : {}".format(len(files_train)))
logging.info("Test files : {}".format(len(files_test)))


###########################################################################
logging.info(" - Compute a PCA on the dataset : unsupervised feature extraction /dimensionality reduction")
###########################################################################

pca = RandomizedPCA(n_components=n_components, whiten=True)

start = time()

logging.info("-- setup train data")

# t0 = time()
# X_train_, width_, height_ = get_data(files_train, resize_factor)
# print "get_data : ", time() - t0
# t0 = time()
X_train, width, height = get_data_parallel(files_train, resize_factor)
# print "get_data_parallel : ", time() - t0
# assert width == width_ and height == height_ and (X_train == X_train_).all(), "Data is not identical"

logging.info("-- setup test data")

# t0 = time()
# X_test_, width_, height_ = get_data(files_test, resize_factor)
# print "get_data : ", time() - t0
# t0 = time()
X_test, width__, height__ = get_data_parallel(files_test, resize_factor)
# print "get_data_parallel : ", time() - t0
# assert width__ == width_ and height__ == height_ and (X_test == X_test).all(), "Data is not identical"


# Compute PCA
logging.info("-- PCA fit")
pca.fit(X_train)

logging.info("Elapsed seconds : {}".format(time() - start))

# Save trained pca model :
joblib.dump(pca, os.path.join('models', pca_filename))

eigenposes = pca.components_.reshape((n_components, height, width))

logging.info("Project the input data on the eigenposes orthogonal basis")
start = time()

X_train_pca = pca.transform(X_train)
X_test_pca = pca.transform(X_test)

logging.info("Elapsed seconds : {}".format(time() - start))


###############################################################################
logging.info("Train a SVM classification model")
###############################################################################

start = time()

param_grid = {'C': [1e3, 5e3, 1e4, 5e4, 1e5],
              'gamma': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1], }
clf = GridSearchCV(SVC(kernel='rbf', class_weight='balanced', probability=True), param_grid)
clf = clf.fit(X_train_pca, target_train)

logging.info("Elapsed seconds : {}".format(time() - start))

# Save trained classifier :
joblib.dump(clf, os.path.join('models', model_filename))

print("Best estimator found by grid search:")
print(clf.best_estimator_)


###############################################################################
logging.info("Quantitative evaluation of the model quality on the test set")
###############################################################################

start = time()
target_pred = clf.predict(X_test_pca)
logging.info("Elapsed seconds : {}".format(time() - start))


cr = classification_report(target_test, target_pred)
print cr
cm = confusion_matrix(target_test, target_pred, labels=range(n_classes))
print cm

target_proba_pred = clf.predict_proba(X_test_pca)

score = log_loss(target_test, target_proba_pred)
print 'Score log_loss: ', score



# Write out metrics:
metrics_output = 'training_' + str(len(classes)) + '_' \
                + str(len(classes) * n_samples_per_class) + '_' \
                + str(n_components) + '_' \
                + str(resize_factor) \
                + '.log'

with open(os.path.join('results', metrics_output), 'w') as writer:
    writer.writelines(cr)
    writer.write('\n\n')
    writer.write(str(cm))
    writer.write('\n\n')
    writer.write('Score log_loss: {}'.format(score))

