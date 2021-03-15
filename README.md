# Computer Vision Duth
A repository containing all the projects done during the 7th semester university subject "Computer Vision"
The code is written in Python mostly using **OpenCV** (and for the final project **Keras**)

## Project 1 - Object Counting from a microscope image
In this project we must:
Given 2 same images from a microscope (One with added Salt and Pepper Noise)
-   Create a Median filter in order to remove the noise from the second image 
-   Count the number of cells that are inside the image boundaries .
-   Count the area (in pixels) of each cell
-   Calculate the average grayscale value for the pixels inside the bounding box of each cell, using a method with constant time complexity (**Integral Image**)
## Project 2 - Image Stitching

In this project we create a **Panoramic photo** by stitching 4 photos, by extracting their features using the 
**SURF** and **SIFT** algorithms

## Project 3 - Image Classification using Bag of Visual Words Model 
In this project we classify part of the **Caltech-256** Dataset  by training a BoV model using **k-means**
and then using the following classifiers:
- Support Vector Machines (One vs all)
- K nearest neighbor   

For all methods many hyperparameter comparisons were made

## Project 4 - Image Classification using Convolutional Neural Netwroks (CNN)
In the final project we classify the same dataset of the previous one using 
- Different **Deep Neural Network** architectures
- Data Augmentations
- Transfer Learning
