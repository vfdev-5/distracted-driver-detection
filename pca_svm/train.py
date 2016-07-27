#
# Script to train SVM Classifier on PCA reduced data
#

# Python
import logging
import os
from time import time

# Sklearn
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import log_loss
from sklearn.grid_search import GridSearchCV
from sklearn.decomposition import RandomizedPCA
from sklearn.svm import SVC
from sklearn.externals import joblib

# Project
from common.preprocessing import get_data_parallel
from common.datasets import get_drivers_list, trainval_files

logger = logging.getLogger()
logger.setLevel(logging.INFO)

###########################################################################
# DEFINE GLOBAL TRAINING PARAMETERS
###########################################################################

classes = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9)
# number of images per class and per driver
nb_samples = 20

# all drivers
all_drivers = get_drivers_list()
drivers = all_drivers

# Validation size
validation_size = 0.4

# Image resize factor
resize_factor = 8

# Number of components for PCA
n_components = 150


###########################################################################
# Check for existing result
###########################################################################
n_classes = len(classes)

model_filename = 'svc_' + str(n_classes) + '_' \
                 + str(n_classes * nb_samples * len(drivers)) + '_' \
                 + str(int(validation_size*100)) + '_' \
                 + str(n_components) + '_' \
                 + str(resize_factor) \
                 + '.pkl'

pca_filename = 'pca_' + str(n_classes) + '_' \
               + str(n_classes * nb_samples * len(drivers)) + '_' \
               + str(int(validation_size*100)) + '_' \
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

train_files, train_targets, test_files, test_targets = trainval_files(classes, all_drivers, nb_samples, validation_size)

logging.info("Train files : {}".format(len(train_files)))
logging.info("Test files : {}".format(len(test_files)))


###########################################################################
logging.info(" - Compute a PCA on the dataset : unsupervised feature extraction /dimensionality reduction")
###########################################################################

pca = RandomizedPCA(n_components=n_components, whiten=True)

start = time()

logging.info("-- setup train data")
X_train, width, height = get_data_parallel(train_files, resize_factor)
logging.info("-- setup test data")
X_test, _, _ = get_data_parallel(test_files, resize_factor)

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
clf = clf.fit(X_train_pca, train_targets)

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


cr = classification_report(test_targets, target_pred)
print cr
cm = confusion_matrix(test_targets, target_pred, labels=range(n_classes))
print cm

target_proba_pred = clf.predict_proba(X_test_pca)

score = log_loss(test_targets, target_proba_pred)
print 'Score log_loss: ', score


# Write out metrics:
metrics_output = 'training_' + str(n_classes) + '_' \
                 + str(n_classes * nb_samples * len(drivers)) + '_' \
                 + str(int(validation_size*100)) + '_' \
                 + str(n_components) + '_' \
                 + str(resize_factor) \
                 + '.log'

with open(os.path.join('results', metrics_output), 'w') as writer:
    writer.writelines(cr)
    writer.write('\n\n')
    writer.write(str(cm))
    writer.write('\n\n')
    writer.write('Score log_loss: {}'.format(score))

