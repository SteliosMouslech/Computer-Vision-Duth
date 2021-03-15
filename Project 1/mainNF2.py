import cv2 as cv2
import numpy as np

final = cv2.imread("NF2.png", cv2.IMREAD_GRAYSCALE)

cv2.imshow('source', final)  # Show the image
cv2.waitKey(0)

thresh, final_Binary= cv2.threshold(final, 53, 255, cv2.THRESH_BINARY)

cv2.imshow('Final_Picture1_Binary', final_Binary)  # Show the image
cv2.waitKey(0)

kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
final_transformed = cv2.morphologyEx(final_Binary, cv2.MORPH_CLOSE, kernel,iterations=1)
final_transformed = cv2.morphologyEx(final_transformed, cv2.MORPH_OPEN, kernel,iterations=5)
cv2.imshow('Final_Picture12', final_transformed)  # Show the image
cv2.waitKey(0)


_, contours, _ = cv2.findContours(final_transformed, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

print("Number of Cells in the image including those touching the borders " + str(len(contours)))

final_copy=final.copy()
cv2.drawContours(final_copy, contours, -1, (0,255,0), 3)
cv2.imshow('contours',final_copy)
cv2.waitKey(0)


def isContourAtBorder(contour,image):
    x, y, w, h = cv2.boundingRect(contour)
    xMin = 0
    yMin = 0
    xMax = image.shape[1]-1
    yMax = image.shape[0]-1
    if x <= xMin or y <= yMin or x+w >= xMax or y+h >= yMax:
        return True
    else:
        return False


#find the bordering contours
badconindex = np.array([])
i=0
for c in contours:
    if isContourAtBorder(c,final_transformed):
        badconindex = np.append(badconindex,i)
    i = i + 1
contours = np.delete(contours, badconindex)


print("Number of Cells in the image without those touching the borders " + str(len(contours)))

final_copy2=final.copy()

def countAreaContour(contour):
    x, y, w, h = cv2.boundingRect(contour)
    size=0
    for y1 in range(y,y+h):
        for x1 in range(x,x+w):
            if cv2.pointPolygonTest(contour, (x1,y1),False)>= 0:
                if final_transformed[y1][x1]==255:
                    size=size+1
    return size

i=0
for c in contours:
    final_labeled = cv2.putText(final_copy2, str(i), cv2.boundingRect(contours[i])[:2], cv2.FONT_HERSHEY_COMPLEX, 1, [255])
    area = cv2.contourArea(c)
    print("Area of cell number (using ccntourArea) ",i," is", area)
    print("Area of cell number   (using my method) ",i," is", countAreaContour(c))

    i = i + 1
#cv2.drawContours(final_labeled, contours, -1, (0,255,0), 3)
cv2.imshow('contours',final_labeled)
cv2.waitKey(0)



integral_img=cv2.integral(final,cv2.CV_64F)
i=0
for c in contours:
    x, y, w, h = cv2.boundingRect(c)
    sumOfPixels=integral_img[y][x] + integral_img[y+h][x+w] - integral_img[y][x+w] - integral_img[y+h][x]
    meanGrayValue = sumOfPixels/(h*w)
    print("Mean Gray Value of bounding box of  Cell number ",i, "is ", meanGrayValue)
    i=i+1
cv2.waitKey(0)



