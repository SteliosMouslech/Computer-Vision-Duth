import cv2 as cv
import numpy as np


vocabulary = np.load('vocabulary.npy')
test_bow_descs = np.load('test_bow_descs.npy')
test_class_labels = np.load('test_class_labels.npy')

#K-Nearest Neighbors

training_data = np.load('bow_descs.npy')
training_class_labels = np.load('train_class_labels.npy')
K = 20
counter=0
for i in range(test_class_labels.shape[0]):
    sum_labels = np.zeros(6, dtype=int)
    distances = test_bow_descs[i] - training_data
    distances = np.abs(distances)
    distances = np.sum(distances, axis=1)
    sorted_ids = np.argsort(distances)
    for j in range (K):
        sum_labels[training_class_labels[sorted_ids[j]]-1]+= 1
    max_label = np.argmax(sum_labels, axis=0)
    print("Predictied Class(K-nn) : ",max_label+1," Real Class: ",test_class_labels[i])
    if max_label+1 == test_class_labels[i]:
        counter+=1

precision = float(counter)/float(test_class_labels.shape[0])
print("The number of Correct  Matches Using K-nn is ",counter,"out of ",test_class_labels.shape[0]," pictures")
print("The accuracy of the K-nn model is {a:.5f}%".format(a=precision*100))


#SVM

def Image_Classif_SVM(bow_d, svm_list):
    min_prediction= 9999999999999999999
    min_svm= -1
    bow_d = np.expand_dims(bow_d, axis=1)
    bow_d = np.transpose(bow_d)
    for svm in svm_list:
        prediction = svm.predict(bow_d.astype(np.float32), flags=cv.ml.STAT_MODEL_RAW_OUTPUT)[1]
        if prediction[0] <= min_prediction:
            min_prediction = prediction[0]
            min_svm=svm_list.index(svm)+1
    return min_svm


list_of_SVMs = []
for i in range (1, 7):
    SVM_file_name = "SVM" + str(i)
    test_svm = cv.ml.SVM_load(SVM_file_name)
    list_of_SVMs.append(test_svm)


counter=0
for i in range (test_bow_descs.shape[0]):
    classif_label=Image_Classif_SVM(test_bow_descs[i], list_of_SVMs)
    print("Predictied Class(SVM) : ",str(classif_label)," Real Class: ",str(test_class_labels[i]))
    if classif_label == test_class_labels[i] :
        counter = counter+1

precision = float(counter)/float(test_class_labels.shape[0])
print("The number of Correct  Matches Using SVM is ",counter,"out of ",test_class_labels.shape[0]," pictures")
print("The accuracy of the SVM model is {a:.5f}%".format(a=precision*100))
