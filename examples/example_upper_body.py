
#
# Example : apply opencv Haar cascade upper body detector
#

import os
import time

import numpy as np

import matplotlib.pyplot as plt

import cv2

# import classifier
body_haar_cascade_path = "/usr/share/opencv/haarcascades/haarcascade_upperbody.xml"
face_haar_cascade_path = "/usr/share/opencv/haarcascades/haarcascade_profileface.xml"

cascade_body = cv2.CascadeClassifier(body_haar_cascade_path)
cascade_face = cv2.CascadeClassifier(face_haar_cascade_path)


def detectCascade(cascade, gray, scale_factor=1.1, min_neighbors=2, min_size=(30, 30)):
    return cascade.detectMultiScale(gray, scale_factor, min_neighbors, cv2.CASCADE_SCALE_IMAGE, minSize=min_size)



train_files_path = "../input/train/c1"
counter = 1
for f in os.listdir(train_files_path):

    image = cv2.imread(os.path.join(train_files_path, f))
    # image = cv2.resize(image, (0, 0), fx=0.7, fy=0.7, interpolation=cv2.INTER_LINEAR)
    start = time.time()

    # RGB -> GRAY -> EqualizeHist
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    gray = cv2.equalizeHist(gray)

    # Detect with cascades :
    bodies = detectCascade(cascade_body, gray, min_size=(150, 150))
    faces = detectCascade(cascade_face, gray)

    print "Elapsed seconds : ", time.time() - start
    print "BODIES:", bodies
    print "FACES:", faces

    # Draw rectangles :
    for (x, y, w, h) in bodies:
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 255), 1)


    # Show the output image :
    cv2.imshow("Detections", image)
    cv2.waitKey(0)

    if counter < 0:
        break
    counter -= 1