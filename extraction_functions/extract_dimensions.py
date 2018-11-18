import cv2
import numpy as np 
from matplotlib import pyplot as plt

def extract_dimensions(img):
    # threshold image
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    ret, threshed_img = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    # find contours
    image, contours, hier = cv2.findContours(threshed_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # with each contour, draw boundingRect in green
    # a min_area_rect in blue
    min_area_rect = []
    boxes = []

    for i, c in enumerate(contours):
        # get the bounding rect
        x, y, w, h = cv2.boundingRect(c)
        # draw a green rectangle to visualize the bounding rect
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
    
        # get the min area rect
        rect = cv2.minAreaRect(c)
        box = cv2.boxPoints(rect)
        # convert all coordinates floating point values to int
        box = np.int0(box)
        min_area_rect = box
        boxes.append((i,box,w,h))
        # draw a blue 'nghien' rectangle
        cv2.drawContours(img, [box], 0, (0, 0, 255), 3)

    #order all the contours by area from biggest to smallest
    boxes.sort(key=lambda x: x[2]*x[3], reverse=True)

    #now we want to select the second biggest box (since the biggest one is the contour of the whole image)
    min_area_rectangle = boxes[1][1]
    min_area_rectangleIndex = boxes[1][0]

    #get x and y points and width/height of the countour we selected
    x,y,w,h = cv2.boundingRect(contours[min_area_rectangleIndex])
    
    if w > h:
        temp = w
        w = h
        h = temp

    return [w, h]