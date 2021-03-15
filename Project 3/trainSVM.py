import os
import cv2 as cv
import numpy as np

training_data = np.load('bow_descs.npy')
training_class_labels = np.load('train_class_labels.npy')


for i in range (1,7):
    current_labels = np.zeros((training_class_labels.shape),dtype=int)
    n1 = training_class_labels.shape[0]
    for j in range (n1):
        if training_class_labels[j] == i:
            current_labels[j] = 1;
    svm = cv.ml.SVM_create()
    svm.setType(cv.ml.SVM_C_SVC)
    svm.setKernel(cv.ml.SVM_RBF)
    svm.setTermCriteria((cv.TERM_CRITERIA_MAX_ITER, 100, 1e-6))
    svm.trainAuto(training_data.astype(np.float32), cv.ml.ROW_SAMPLE, current_labels)
    SVM_file_name = "SVM" + str(i)
    svm.save(SVM_file_name)
    print (SVM_file_name)