import numpy as np
import cv2
from collections import deque
import os
import numpy as np 
import pandas as pd 
import cv2
import tensorflow as tf
from keras.models import load_model
import matplotlib.pyplot as plt
import imutils
from imutils.contours import sort_contours
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Dense
from keras.layers import Flatten
from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator
from keras.models import model_from_json
import os

chars = []
labels = ['%', '*', '+', '-', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '[', ']']

image = cv2.imread('C:\\Users\\DC\\OneDrive\\Desktop\\computer-vision-project\\screenshots\\newImage.png')


def prediction(img):
    #img = cv2.imread(img, cv2.IMREAD_GRAYSCALE)
    plt.imshow(img, cmap = 'gray')
    img = cv2.resize(img,(40, 40))
    norm_image = cv2.normalize(img, None, alpha = 0, beta = 1, norm_type = cv2.NORM_MINMAX, dtype = cv2.CV_32F)
    #norm_image=img/255
    norm_image = norm_image.reshape((norm_image.shape[0], norm_image.shape[1], 1))
    case = np.asarray([norm_image])
    pred = model.predict([case])
    return(np.argmax(pred))


with open('my_model_architecture.json', 'r') as f:
    model_json = f.read()
model = model_from_json(model_json)

# Load model weights from file
model.load_weights('my_model_weights.h5')

def function():
    image = cv2.imread('C:\\Users\\DC\\OneDrive\\Desktop\\opencv_project\\screenshots\\newimg.png')
  #image = cv2.resize(image,(300,300))
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
  # perform edge detection, find contours in the edge map, and sort the
  # resulting contours from left-to-right
    edged = cv2.Canny(blurred, 30, 150)
    cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    cnts = sort_contours(cnts, method="left-to-right")[0]
    for c in cnts:
      # compute the bounding box of the contour
        (x, y, w, h) = cv2.boundingRect(c)
      # filter out bounding boxes, ensuring they are neither too small
      # nor too large
        if w*h>1200:
          # extract the character and threshold it to make the character
          # appear as *white* (foreground) on a *black* background, then
          # grab the width and height of the thresholded image
            roi = gray[y:y + h, x:x + w]
            chars.append(prediction(roi))
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            plt.figure(figsize=(10,10))
            plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            plt.savefig('image.png')

def function_other():
    
    # plt.imshow(image)
    
    eq=""
    for i in chars:
      eq += str(labels[i])
    print(eq)


function()
function_other()