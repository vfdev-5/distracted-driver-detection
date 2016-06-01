
#
# Example : create opencv bag of words
#

import os
import time

import numpy as np

import cv2

train_files_path = "../input/train/c1"

feature_detectors = [
    cv2.FastFeatureDetector()
    # cv2.Feature2D_create("BRISK"),
    # cv2.Feature2D_create("ORB"),
]

descriptor_extractor = cv2.DescriptorExtractor_create("ORB")

nb_clusters = 10
bow = cv2.BOWKMeansTrainer(nb_clusters)

counter = 5
for f in os.listdir(train_files_path):

    image = cv2.imread(os.path.join(train_files_path, f))

    # Start image preprocessing:
    start = time.time()
    image = cv2.resize(image, (0, 0), fx=0.5, fy=0.5, interpolation=cv2.INTER_LINEAR)
    proc = cv2.medianBlur(image, 3)

    # Compute mat of descriptors :
    proc_with_kp = proc.copy()
    for detector in feature_detectors:
        kp = detector.detect(proc)
        kp, des = descriptor_extractor.compute(proc, kp)
        print des.shape, des.dtype
        proc_with_kp = cv2.drawKeypoints(proc_with_kp, kp, color=(0, 255, 0))
        bow.add(des.astype(np.float32))

    print "Elapsed seconds : ", time.time() - start

    # Show the output image :
    cv2.imshow("Original", image)
    cv2.imshow("Proc with keypoints", proc_with_kp)
    cv2.waitKey(0)

    if counter <= 0:
        break
    counter -= 1


voc = bow.cluster()
print voc.shape, voc.dtype
print voc
