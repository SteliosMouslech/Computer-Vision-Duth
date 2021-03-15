import os
import cv2 as cv
import numpy as np


imagedb_train = ('imagedb_train')

sift = cv.xfeatures2d_SIFT.create()
folders = os.listdir(imagedb_train)

def extract_local_features(path):
    img = cv.imread(path)

    kp = sift.detect(img)
    desc = sift.compute(img, kp)
    desc = desc[1]
    return desc


train_descs = np.zeros((0, 128), dtype=np.float32)

for folder in folders:
    current_folder = os.path.join(imagedb_train, folder)
    print(current_folder)  #gia na vlepw se poio simio ine o kodikas
    files = os.listdir(current_folder)
    for file in files:
        current_file = os.path.join(current_folder, file)
        desc = extract_local_features(current_file)
        train_descs = np.concatenate((train_descs, desc), axis=0)

# Create vocabulary
print("K-Means Calculating...")  #gia na vlepw se poio simio ine o kodikas

term_crit = (cv.TERM_CRITERIA_EPS, 50, 0.1)
loss, assignments, vocabulary = cv.kmeans(train_descs.astype(np.float32), 50, None, term_crit, 1, 0)
np.save('vocabulary.npy', vocabulary)

#vocabulary = np.load('vocabulary.npy')


# Create Histograms
current_label = np.full((1, 1), 1, dtype=int)
train_class_labels = np.zeros((0, 1), dtype=int)


def getBovwDescriptor(desc, vocabulary):
    bow_desc = np.zeros((1, vocabulary.shape[0]), dtype=np.float32)
    for d in range(desc.shape[0]):
        distances = desc[d, :] - vocabulary
        distances = np.abs(distances)
        distances = np.sum(distances, axis=1)
        mini = np.argmin(distances)
        bow_desc[0, mini] += 1
    return bow_desc


bow_descs = np.zeros((0, vocabulary.shape[0]), dtype=np.float32)
for folder in folders:
    current_folder = os.path.join(imagedb_train, folder)
    print(current_folder)
    files = os.listdir(current_folder)
    for file in files:
        current_file = os.path.join(current_folder, file)
        print(current_file)
        desc = extract_local_features(current_file)
        bow_desc = getBovwDescriptor(desc, vocabulary)
        bow_descs = np.concatenate((bow_descs, bow_desc), axis=0)
        train_class_labels=np.concatenate((train_class_labels,current_label), axis=0)
    current_label[0] = current_label[0] + 1

np.save('bow_descs.npy', bow_descs)
np.save('train_class_labels.npy', train_class_labels)

