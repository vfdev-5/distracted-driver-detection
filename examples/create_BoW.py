
#
# Example : create opencv bag of words
#

import os
import time

import numpy as np

import cv2


train_files_path = "../input/train/c1"

voc_file_path = "voc"

feature_detectors = [
    cv2.FastFeatureDetector()
    # cv2.Feature2D_create("BRISK"),
    # cv2.Feature2D_create("ORB"),
]

descriptor_extractor = cv2.DescriptorExtractor_create("ORB")
descriptor_matcher = cv2.DescriptorMatcher_create("BruteForce-Hamming")

if not os.path.exists(voc_file_path):

    nb_clusters = 20
    bow = cv2.BOWKMeansTrainer(nb_clusters)
    counter = 10
    for f in os.listdir(train_files_path):

        image = cv2.imread(os.path.join(train_files_path, f))

        # Start image preprocessing:
        start = time.time()
        image = cv2.resize(image, (0, 0), fx=0.5, fy=0.5, interpolation=cv2.INTER_LINEAR)
        proc = cv2.medianBlur(image, 3)

        # Compute mat of descriptors :
        # proc_with_kp = proc.copy()
        for detector in feature_detectors:
            kp = detector.detect(proc)
            kp, des = descriptor_extractor.compute(proc, kp)
            print des.shape, des.dtype
            # proc_with_kp = cv2.drawKeypoints(proc_with_kp, kp, color=(0, 255, 0))
            bow.add(des.astype(np.float32))

        print "Elapsed seconds : ", time.time() - start

        # Show the output image :
        # cv2.imshow("Original", image)
        # cv2.imshow("Proc with keypoints", proc_with_kp)
        # cv2.waitKey(0)

        if counter <= 0:
            break
        counter -= 1

    voc = bow.cluster()
    print "Voc: ", voc.shape, voc.dtype
    # print voc
    np.save(voc_file_path, voc)

else:
    voc = np.load(voc_file_path)

bowIde = cv2.BOWImgDescriptorExtractor(dextractor=descriptor_extractor, dmatcher=descriptor_matcher)
bowIde.setVocabulary(voc.astype(np.uint8))

def get_roi(callback):

    # https://books.google.fr/books?id=9uVOCwAAQBAJ&pg=PA60&lpg=PA60&dq=python+opencv+mouse+rectangle&source=bl&ots=uzzMDtY2EV&sig=FFWjeyu1P-NEDUa1cqxhmY-exiQ&hl=en&sa=X&ved=0ahUKEwjssejF94jNAhUKWBoKHXsVDSgQ6AEIQDAG#v=onepage&q=python%20opencv%20mouse%20rectangle&f=false
    def _get_roi(event, x, y, flags, params):
        global x_init, y_init, drawing, tl_pt, br_pt

        # Detecting a mouse click
        if event == cv2.EVENT_LBUTTONDOWN:
            drawing = True
            x_init, y_init = x, y
        # Detecting mouse move event
        elif event == cv2.EVENT_MOUSEMOVE:
            if drawing:
                tl_pt, br_pt = (x_init, y_init), (x, y)
                # cv2.rectangle(img_orig, tl_pt, br_pt, (0, 0, 255), 2)
        elif event == cv2.EVENT_LBUTTONUP:
            drawing = False
            tl_pt, br_pt = (x_init, y_init), (x, y)
            # cv2.rectangle(img_orig, tl_pt, br_pt, (0, 0, 255), 2)
            rect_final = (x_init, y_init, x, y)
            # Do something with
            callback(rect_final)

    return _get_roi

train_files_path = "../input/train/c1"
counter = 5
for f in os.listdir(train_files_path):
    image = cv2.imread(os.path.join(train_files_path, f))

    # Start image preprocessing:
    image = cv2.resize(image, (0, 0), fx=0.5, fy=0.5, interpolation=cv2.INTER_LINEAR)
    proc = cv2.medianBlur(image, 3)
    print proc.shape

    def compute_words(rect):
        print "Compute words with roi", rect
        roi = proc[rect[1]:rect[3], rect[0]:rect[2], :]
        # Compute mat of descriptors :
        for detector in feature_detectors:
            kp = detector.detect(roi)
            kp, des = descriptor_extractor.compute(roi, kp)

        print len(kp)
        if len(kp) > 0:
            hist = bowIde.compute(roi, kp)
            print np.argmax(hist, 1), hist
        roi_with_kp = roi.copy()
        roi_with_kp = cv2.drawKeypoints(roi_with_kp, kp, color=(0, 255, 0))
        cv2.imshow("ROI with keypoints", roi_with_kp)


    # Draw rectangle UI
    img_orig = proc.copy()
    drawing = False
    cv2.namedWindow("Input")
    cv2.setMouseCallback("Input", get_roi(compute_words))

    # Show the output image :
    cv2.imshow("Input", img_orig)
    cv2.waitKey(0)

    if counter <= 0:
        break
    counter -= 1

