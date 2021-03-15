import os
import cv2 as cv
import numpy as np

imagedb_test = ('imagedb_test')

sift = cv.xfeatures2d_SIFT.create()
folders = os.listdir(imagedb_test)
current_class_label = np.full((1, 1), 1, dtype=int)
test_class_labels = np.zeros((0, 1), dtype=int)

def extract_local_features(path):
    img = cv.imread(path)

    kp = sift.detect(img)
    desc = sift.compute(img, kp)
    desc = desc[1]
    return desc

def getBovwDescriptor(desc, vocabulary):
    bow_desc = np.zeros((1, vocabulary.shape[0]), dtype=np.float32)
    for d in range(desc.shape[0]):
        distances = desc[d, :] - vocabulary
        distances = np.abs(distances)
        distances = np.sum(distances, axis=1)
        mini = np.argmin(distances)
        bow_desc[0, mini] += 1
    return bow_desc

vocabulary = np.load('vocabulary.npy')

# Create class labels


train_descs = np.zeros((0, 128), dtype=np.float32)
bow_descs = np.zeros((0, vocabulary.shape[0]), dtype=np.float32)
for folder in folders:
    current_folder = os.path.join(imagedb_test, folder)
    print(current_folder)
    files = os.listdir(current_folder)
    for file in files:
        current_file = os.path.join(current_folder, file)
        print(current_file)
        desc = extract_local_features(current_file)
        bow_desc = getBovwDescriptor(desc, vocabulary)
        bow_descs = np.concatenate((bow_descs, bow_desc), axis=0)
        test_class_labels=np.concatenate((test_class_labels,current_class_label), axis=0)
    current_class_label[0] = current_class_label[0] + 1

np.save('test_bow_descs.npy', bow_descs)
np.save('test_class_labels.npy', test_class_labels)