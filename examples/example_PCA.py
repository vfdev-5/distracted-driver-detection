
#
# Example of PCA + SVM usage on training data
#

# Python
import os
import logging
from time import time

# Numpy & Matplotlib
import numpy as np
import matplotlib.pyplot as plt

# Sklearn
from sklearn.cross_validation import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.grid_search import GridSearchCV
from sklearn.decomposition import RandomizedPCA
from sklearn.svm import SVC
from sklearn.externals import joblib

# Opencv
import cv2


logger = logging.getLogger()
logger.setLevel(logging.INFO)

###########################################################################
logging.info("- Prepare training data")
###########################################################################

classes = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9)
n_samples_per_class = 50
targets = []
files = []

for cls in classes:
    path = os.path.join("../input/train", 'c' + str(cls))
    counter = n_samples_per_class
    for f in os.listdir(path):
        files.append(os.path.join(path, f))
        targets.append(cls)
        counter -= 1
        if counter == 0:
            break

# print len(files)
# print files[8:12]
# print targets[8:12]

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


n_components = 100
pca = RandomizedPCA(n_components=n_components, whiten=True)

start = time()


def preprocess_image(in_image, w, h):
    gray = cv2.cvtColor(in_image, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    gray = cv2.resize(gray, (w, h))
    return gray.reshape(w*h)


def get_data(files):
    out = None
    w = None
    h = None
    print "Total files : ", len(files)
    for i, f in enumerate(files):
        image = cv2.imread(f)
        if out is None:
            w, h = image.shape[1] / 8, image.shape[0] / 8
            out = np.empty((len(files), w * h))
        data = preprocess_image(image, w, h)
        out[i, :] = data[:]

        if i % 21 == 0:
            print i,
        elif i % 200 == 0:
            print i
    return out, w, h

logging.info("-- setup train data")
X_train, width, height = get_data(files_train)
logging.info("-- setup test data")
X_test, _, _ = get_data(files_test)
logging.info("-- setup data is done")

# Compute PCA
pca.fit(X_train)

logging.info("Elapsed seconds : {}".format(time() - start))

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
clf = GridSearchCV(SVC(kernel='rbf', class_weight='balanced'), param_grid)
clf = clf.fit(X_train_pca, target_train)

logging.info("Elapsed seconds : {}".format(time() - start))

# Save trained classifier :
if not os.path.exists('models'):
    os.mkdir('models/')
filename = 'models/svc_' + str(n_classes) + '_' + str(len(target_train)) + '.pkl'
joblib.dump(clf, filename)

print("Best estimator found by grid search:")
print(clf.best_estimator_)

#
# THIS GIVES :
#
#     precision    recall  f1-score   support
#
# 0       1.00      0.89      0.94        19
# 1       0.87      0.72      0.79        18
# 2       0.74      0.94      0.83        18
# 3       0.85      0.85      0.85        20
#
# avg / total       0.87      0.85      0.85        75
#
# [[17  1  1  0]
#  [ 0 13  2  3]
#  [ 0  1 17  0]
#  [ 0  0  3 17]]


#clf = SVC()
#clf.fit(X_train_pca, target_train)
#
# THIS GIVES :
#
#       precision    recall  f1-score   support
#
# 0       1.00      0.84      0.91        19
# 1       0.88      0.78      0.82        18
# 2       0.77      0.94      0.85        18
# 3       0.86      0.90      0.88        20
#
# avg / total       0.88      0.87      0.87        75
#
# [[16  1  1  1]
#  [ 0 14  2  2]
#  [ 0  1 17  0]
#  [ 0  0  2 18]]


###############################################################################
logging.info("Quantitative evaluation of the model quality on the test set")

start = time()
target_pred = clf.predict(X_test_pca)
logging.info("Elapsed seconds : {}".format(time() - start))

print classification_report(target_test, target_pred)
print confusion_matrix(target_test, target_pred, labels=range(n_classes))

###############################################################################
logging.info("Qualitative evaluation of the predictions using matplotlib")
###############################################################################


def plot_gallery(images, titles, h, w, n_row=3, n_col=4):
    """Helper function to plot a gallery of portraits"""
    plt.figure(figsize=(1.8 * n_col, 2.4 * n_row))
    plt.subplots_adjust(bottom=0, left=.01, right=.99, top=.90, hspace=.35)
    for i in range(n_row * n_col):
        plt.subplot(n_row, n_col, i + 1)
        plt.imshow(images[i].reshape((h, w)), cmap=plt.cm.gray)
        plt.title(titles[i], size=12)
        plt.xticks(())
        plt.yticks(())


# plot the result of the prediction on a portion of the test set
def title(y_pred, y_test, target_names, i):
    pred_name = target_names[y_pred[i]]
    true_name = target_names[y_test[i]]
    return 'predicted: %s\ntrue:      %s' % (pred_name, true_name)

prediction_titles = [title(target_pred, target_test, range(n_classes), i)
                      for i in range(target_pred.shape[0])]
plot_gallery(X_test, prediction_titles, height, width)

# plot the gallery of the most significative eigenfaces
eigenpose_titles = ["eigenpose %d" % i for i in range(eigenposes.shape[0])]
plot_gallery(eigenposes, eigenpose_titles, height, width)

plt.show()

"""

Precision = (true positives) / ((true positives) + (false positives))
Recall = (true positives) / ((true positives) + (false negatives))
F1-score = 2* (Precision * Recall)/(Precision + Recall)


Results for 9 classes with

75 images per class

Best estimator found by grid search:
SVC(C=1000.0, cache_size=200, class_weight='balanced', coef0=0.0,
  decision_function_shape=None, degree=3, gamma=0.01, kernel='rbf',
  max_iter=-1, probability=False, random_state=None, shrinking=True,
  tol=0.001, verbose=False)

precision    recall  f1-score   support

0       0.69      0.80      0.74        25
1       0.81      0.81      0.81        21
2       0.89      0.89      0.89        19
3       0.67      0.93      0.78        15
4       0.90      0.83      0.86        23
5       1.00      0.94      0.97        17
6       0.72      0.72      0.72        18
7       1.00      0.75      0.86        12
8       1.00      0.84      0.91        19
9       0.78      0.74      0.76        19

avg / total       0.84      0.82      0.83       188

[[20  2  1  2  0  0  0  0  0  0]
 [ 2 17  0  1  0  0  1  0  0  0]
 [ 0  0 17  0  1  0  1  0  0  0]
 [ 0  0  0 14  1  0  0  0  0  0]
 [ 0  0  0  3 19  0  1  0  0  0]
 [ 1  0  0  0  0 16  0  0  0  0]
 [ 1  1  1  1  0  0 13  0  0  1]
 [ 1  0  0  0  0  0  0  9  0  2]
 [ 0  0  0  0  0  0  2  0 16  1]
 [ 4  1  0  0  0  0  0  0  0 14]]



Results for 9 classes with

150 images per class

Best estimator found by grid search:
SVC(C=1000.0, cache_size=200, class_weight='balanced', coef0=0.0,
  decision_function_shape=None, degree=3, gamma=0.01, kernel='rbf',
  max_iter=-1, probability=False, random_state=None, shrinking=True,
  tol=0.001, verbose=False)

precision    recall  f1-score   support

0       0.86      0.97      0.91        33
1       0.95      0.95      0.95        37
2       0.93      1.00      0.96        41
3       0.85      0.97      0.91        36
4       1.00      0.86      0.92        35
5       0.97      0.97      0.97        30
6       0.93      0.97      0.95        39
7       1.00      1.00      1.00        51
8       1.00      0.83      0.91        41
9       0.97      0.91      0.94        32

avg / total       0.95      0.94      0.94       375

[[32  0  0  0  0  1  0  0  0  0]
 [ 1 35  0  1  0  0  0  0  0  0]
 [ 0  0 41  0  0  0  0  0  0  0]
 [ 1  0  0 35  0  0  0  0  0  0]
 [ 0  0  0  5 30  0  0  0  0  0]
 [ 0  0  1  0  0 29  0  0  0  0]
 [ 0  0  1  0  0  0 38  0  0  0]
 [ 0  0  0  0  0  0  0 51  0  0]
 [ 1  1  1  0  0  0  3  0 34  1]
 [ 2  1  0  0  0  0  0  0  0 29]]


Results for 9 classes with

all images per class

Best estimator found by grid search:
SVC(C=1000.0, cache_size=200, class_weight='balanced', coef0=0.0,
  decision_function_shape=None, degree=3, gamma=0.01, kernel='rbf',
  max_iter=-1, probability=False, random_state=None, shrinking=True,
  tol=0.001, verbose=False)

 precision    recall  f1-score   support

0       0.99      1.00      1.00       573
1       1.00      1.00      1.00       605
2       1.00      1.00      1.00       598
3       1.00      0.99      0.99       605
4       0.99      1.00      0.99       589
5       1.00      1.00      1.00       577
6       1.00      1.00      1.00       586
7       1.00      1.00      1.00       480
8       0.99      0.99      0.99       491
9       1.00      0.99      1.00       502

avg / total       1.00      1.00      1.00      5606


[[572   0   0   1   0   0   0   0   0   0]
 [  0 605   0   0   0   0   0   0   0   0]
 [  0   0 597   0   0   0   1   0   0   0]
 [  1   0   2 600   2   0   0   0   0   0]
 [  0   0   0   1 587   0   0   0   1   0]
 [  2   0   0   0   0 575   0   0   0   0]
 [  1   0   0   0   1   0 584   0   0   0]
 [  0   0   0   0   0   0   0 480   0   0]
 [  0   0   0   0   2   0   0   1 487   1]
 [  0   0   0   0   0   0   0   0   3 499]]


"""