
import numpy as np
import cv2 as cv
sift = cv.xfeatures2d_SIFT.create()
img1 = cv.imread('yard-05.png', cv.IMREAD_GRAYSCALE)
img1 = cv.resize(img1,None,fx=0.5,fy=0.5)

kp1 = sift.detect(img1)
desc1 = sift.compute(img1, kp1)


img2 = cv.imread('yard-04.png',cv.IMREAD_GRAYSCALE)
img2 = cv.resize(img2,None,fx=0.5,fy=0.5)

kp2 = sift.detect(img2)
desc2 = sift.compute(img2, kp2)



img3 = cv.imread('yard-03.png',cv.IMREAD_GRAYSCALE)
img3 = cv.resize(img3,None,fx=0.5,fy=0.5)

kp3 = sift.detect(img3)
desc3 = sift.compute(img3, kp3)

img4 = cv.imread('yard-02.png',cv.IMREAD_GRAYSCALE)
img4 = cv.resize(img4,None,fx=0.5,fy=0.5)

kp4 = sift.detect(img4)
desc4 = sift.compute(img4, kp4)


def match(d1, d2):
    n1 = d1.shape[0]
    n2 = d2.shape[0]

    matches = []
    for i in range(n1):
        fv1 = d1[i, :]
        # L2 distance
        i1,mindist2=closestDistanceIndex(fv1,d2)
        fv2= d2[i1, :]
        i2,_=closestDistanceIndex(fv2,d1)

        if i2==i:
            matches.append(cv.DMatch(i, i1, mindist2))

    return matches


def closestDistanceIndex(one_desc1,all_desc2):
    diff = all_desc2 - one_desc1
    diff=np.square(diff)
    distances = np.sum(diff, axis=1)
    distances = np.sqrt(distances)
    index = np.argmin(distances)
    mindistance = distances[index]
    return index , mindistance

matches1 = match(desc1[1], desc2[1]) #Kαλούμε τη συνάρτηση match
img_pt1 = np.array([kp1[x.queryIdx].pt for x in matches1])
img_pt2 = np.array([kp2[x.trainIdx].pt for x in matches1])
M, mask = cv.findHomography(img_pt2, img_pt1, cv.RANSAC)
# Βρίσκει πώς πρέπει να μετατραπεί η πρώτη για να "ταιριάξει" με τη δευτερη
img5= cv.warpPerspective(img2, M, (img1.shape[1]+500, img1.shape[0]+500))
img5[0: img2.shape[0], 0: img2.shape[1]] = img1

cv.namedWindow('main',cv.WINDOW_NORMAL)
cv.imshow('main', img5)
cv.waitKey(0)


matches2 = match(desc3[1], desc4[1])
img_pt3 = np.array([kp3[x.queryIdx].pt for x in matches2])
img_pt4 = np.array([kp4[x.trainIdx].pt for x in matches2])

M2, mask = cv.findHomography(img_pt4, img_pt3, cv.RANSAC)
# Βρίσκει πώς πρέπει να μετατραπεί η πρώτη για να "ταιριάξει" με τη δευτερη
img6 = cv.warpPerspective(img4, M2, (img3.shape[1]+200, img3.shape[0]+200))
img6 [0: img4.shape[0], 0: img4.shape[1]] = img3
#img4 = cv.resize(img4,None,fx=0.8,fy=0.8)


cv.namedWindow('main2',cv.WINDOW_NORMAL)
cv.imshow('main2', img6)
cv.waitKey(0)



kp6 = sift.detect(img6)
desc6 = sift.compute(img6, kp6)


kp5 = sift.detect(img5)
desc5 = sift.compute(img5, kp5)


matches3 = match(desc6[1], desc5[1])
img_pt6 = np.array([kp6[x.queryIdx].pt for x in matches3])
img_pt5 = np.array([kp5[x.trainIdx].pt for x in matches3])
M3, mask = cv.findHomography(img_pt5, img_pt6, cv.RANSAC)
# Βρίσκει πώς πρέπει να μετατραπει η πρώτη για να "ταιριάξει" με τη δευτερη
img7 = cv.warpPerspective(img5, M3, (img6.shape[1]+500, img6.shape[0]+500))

cv.namedWindow('main3',cv.WINDOW_NORMAL)
cv.imshow('main3', img7)
cv.waitKey(0)

img_x=img6.shape[0]
img_y=img6.shape[1]
img_black_white = np.ones((img_x,img_y), dtype=np.uint8)

for x in range(img_x):
    for y in range(img_y):
        if img6[x, y] == 0:
            img_black_white[x,y] = 0

kernel = np.ones((5,5), np.uint8)
erode = cv.morphologyEx(img_black_white, cv.MORPH_ERODE, kernel)


for x in range(erode.shape[0]):
    for y in range(erode.shape[1]):
        if erode[x, y] != 0:
            img7[x, y] = img6[x, y]

cv.namedWindow('main7',cv.WINDOW_NORMAL)
cv.imshow('main7', img7)

cv.waitKey(0)