import numpy as np
import cv2
from collections import deque
import os
import uuid
import time
import pandas as pd
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
import tkinter as tk
import sympy

def eq_sympy(equation_str):
  try:
      solutions = eval(equation_str)
      print("Solution to this equation:", solutions)
  except :
      print("Sorry, could not solve this equation.")

chars = []
labels = ['%', '*', '+', '-', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '[', ']']

image = cv2.imread('E:\\semWork\\s6\\FCV\\airCanvas\\screenshots\\newImage.png')

def prediction(img):
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
    image = cv2.imread('E:\\semWork\\s6\\FCV\\airCanvas\\screenshots\\newImage.png')
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
      # filter out bounding boxes, ensuring they are neither too small nor too large
        if w*h>1200:
          # extract the character and threshold it to make the character
          # appear as *white* (foreground) on a *black* background, then
          # grab the width and height of the thresholded image
            roi = gray[y:y + h, x:x + w]
            chars.append(prediction(roi))
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            plt.figure(figsize=(10,10))
            plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            plt.savefig('E:\\semWork\\s6\\FCV\\airCanvas\\screenshots\\bounding-box-image.png')
def function_other():
    eq=""
    for i in chars:
      eq += str(labels[i])
    print("The detected equation: ", eq)
    eq_sympy(eq)
    #empty chars
    chars.clear()

# create a directory to save the screenshots
dir_name = 'screenshots'
if not os.path.exists(dir_name):
    os.makedirs(dir_name)

#default called trackbar function 
def setValues(x):
   print("")

# Creating the trackbars needed for adjusting the marker colour
cv2.namedWindow("Color detectors")
cv2.createTrackbar("Upper Hue", "Color detectors", 153, 180,setValues)
cv2.createTrackbar("Upper Saturation", "Color detectors", 255, 255,setValues)
cv2.createTrackbar("Upper Value", "Color detectors", 255, 255,setValues)
cv2.createTrackbar("Lower Hue", "Color detectors", 64, 180,setValues)
cv2.createTrackbar("Lower Saturation", "Color detectors", 72, 255,setValues)
cv2.createTrackbar("Lower Value", "Color detectors", 49, 255,setValues)

# Giving different arrays to handle colour points of different colour
bpoints = [deque(maxlen=1024)]

# These indexes will be used to mark the points in particular arrays of specific colour
black_index = 0

#The kernel to be used for dilation purpose 
kernel = np.ones((5,5),np.uint8)

colors = [(0, 0, 0), (0, 255, 0), (0, 0, 255), (122, 122, 122), (255, 0, 0)]
colorIndex = 0

# Here is code for Canvas setup
paintWindow = np.zeros((471,636,3)) + 255
height, width, _ = paintWindow.shape

# making rectangles in paint window
paintWindow = cv2.rectangle(paintWindow, (40,1), (140,65), colors[2], -1)
paintWindow = cv2.rectangle(paintWindow, (160,1), (255,65), colors[0], -1)
paintWindow = cv2.rectangle(paintWindow, (275,1), (370,65), colors[4], -1)
 
# putting text in paint window
cv2.putText(paintWindow, "CLEAR", (49, 36), cv2.FONT_HERSHEY_PLAIN, 1.5, (255, 255, 255), 2, cv2.LINE_AA)
cv2.putText(paintWindow, "BLACK", (172, 36), cv2.FONT_HERSHEY_PLAIN, 1.5, (255, 255, 255), 2, cv2.LINE_AA)
cv2.putText(paintWindow, "CALCULATE", (277, 36), cv2.FONT_HERSHEY_PLAIN, 1.0, (255, 255, 255), 2, cv2.LINE_AA)

# setting the screen width and height
screen_width = 2000
screen_height = 1500 

# setting the window width and height
window_width = screen_width // 2
window_height = screen_height // 2

# making window resizable
cv2.namedWindow('Paint', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Paint', window_width, window_height)

# Loading the default webcam of PC.
cap = cv2.VideoCapture(0)

# Keep looping
while True:
    # Reading the frame from the camera
    ret, frame = cap.read()

    #Flipping the frame to see same side of yours
    frame = cv2.flip(frame, 1)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Getting the new values of the trackbar in real time as the user changes them
    u_hue = cv2.getTrackbarPos("Upper Hue", "Color detectors")
    u_saturation = cv2.getTrackbarPos("Upper Saturation", "Color detectors")
    u_value = cv2.getTrackbarPos("Upper Value", "Color detectors")
    l_hue = cv2.getTrackbarPos("Lower Hue", "Color detectors")
    l_saturation = cv2.getTrackbarPos("Lower Saturation", "Color detectors")
    l_value = cv2.getTrackbarPos("Lower Value", "Color detectors")
    Upper_hsv = np.array([u_hue,u_saturation,u_value])
    Lower_hsv = np.array([l_hue,l_saturation,l_value])

    # Adding the colour buttons to the live frame for colour access
    frame = cv2.rectangle(frame, (40,1), (140,65), colors[2], -1)
    frame = cv2.rectangle(frame, (160,1), (255,65), colors[0], -1)
    frame = cv2.rectangle(frame, (275,1), (370,65), colors[4], -1)
    
    # Adding labels in the rectangles in the frame
    cv2.putText(frame, "CLEAR", (49, 36), cv2.FONT_HERSHEY_PLAIN, 1.5, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, "BLACK", (172, 36), cv2.FONT_HERSHEY_PLAIN, 1.5, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, "CALCULATE", (277, 36), cv2.FONT_HERSHEY_PLAIN, 1.0, (255, 255, 255), 2, cv2.LINE_AA)

    # Identifying the pointer by making its mask
    Mask = cv2.inRange(hsv, Lower_hsv, Upper_hsv)
    Mask = cv2.erode(Mask, kernel, iterations=15)
    Mask = cv2.morphologyEx(Mask, cv2.MORPH_OPEN, kernel)
    Mask = cv2.dilate(Mask, kernel, iterations=10)

    # Find contours for the pointer after idetifying it
    cnts,_ = cv2.findContours(Mask.copy(), cv2.RETR_EXTERNAL,
    	cv2.CHAIN_APPROX_SIMPLE)
    center = None

    # Ifthe contours are formed
    if len(cnts) > 0:
    	# sorting the contours to find biggest 
        cnt = sorted(cnts, key = cv2.contourArea, reverse = True)[0]
        # Get the radius of the enclosing circle around the found contour
        ((x, y), radius) = cv2.minEnclosingCircle(cnt)
        # Draw the circle around the contour
        cv2.circle(frame, (int(x), int(y)), int(radius), (0, 255, 255), 2)
        # Calculating the center of the detected contour
        M = cv2.moments(cnt)
        center = (int(M['m10'] / M['m00']), int(M['m01'] / M['m00']))

        # Now checking if the user wants to click on any button above the screen 
        if center[1] <= 65:
            if 40 <= center[0] <= 140: # Clear Button
                bpoints = [deque(maxlen=512)]
                black_index = 0
                paintWindow[67:,:,:] = 255
            elif 160 <= center[0] <= 255:
                    colorIndex = 0 # Black
            elif 275 <= center[0] <= 370:
                time.sleep(2)
                screenshot = paintWindow[65:height, 0:width]
                filename = f'newImage.png'
                filepath = os.path.join(dir_name, filename)
                print("screenshot taken")
                print(filepath)
                cv2.imwrite(f"screenshots/newImage.png", screenshot)
                bpoints = [deque(maxlen=512)]#
                black_index = 0
                paintWindow[67:,:,:] = 255  
                function()
                function_other()
        else :
            if colorIndex == 0:
                bpoints[black_index].appendleft(center)
                
    # Append the next deques when nothing is detected to avois messing up
    else:
        bpoints.append(deque(maxlen=512))
        black_index += 1

    # Draw lines of all the colors on the canvas and frame 
    points = [bpoints]
    for i in range(len(points)):
        for j in range(len(points[i])):
            for k in range(1, len(points[i][j])):
                if points[i][j][k - 1] is None or points[i][j][k] is None:
                    continue
                cv2.line(frame, points[i][j][k - 1], points[i][j][k], colors[i], 10)
                cv2.line(paintWindow, points[i][j][k - 1], points[i][j][k], colors[i], 10)

    # Show all the windows
    cv2.imshow("Tracking", frame)
    cv2.imshow("Paint", paintWindow)
    cv2.imshow("mask",Mask)

	# If the 'z' key is pressed then stop the application 
    if cv2.waitKey(1) & 0xFF == ord("z"):
        break

# Release the camera and all resources
cap.release()
cv2.destroyAllWindows()