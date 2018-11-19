import cv2
import numpy as np 
from matplotlib import pyplot as plt

def extract_dimensions(img):
    
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    #converting image to graysclae and find threshold
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 11, 17, 17)
    thresh = cv2.adaptiveThreshold(gray, 80, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    
    #enhance contours by dilating them
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(9,9))
    dilated = cv2.dilate(thresh, kernel)

    # threshold image
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    ret, threshed_img = cv2.threshold(dilated, 127, 255, cv2.THRESH_BINARY)
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

def extract_dimensions_threshold(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    #converting image to graysclae to find threshold
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 11, 17, 17)
    thresh = cv2.adaptiveThreshold(gray, 50, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 11, 2)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(9,9))
    dilated = cv2.dilate(thresh, kernel)

    _, cnts, _  = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    #insert fruit area of original image onto a new image with black background
    mask = np.zeros_like(img)

    #creating new black image same size as the original
    out = np.zeros_like(img)

    #selecting all the pixels from mask that are white and replace them with those same 
    #pixels from the original image (that correspond to the pixels of the fruit)
    out[mask==0] = img[mask==0]

    cv2.fillPoly(mask, pts = cnts, color = (255,255,255))

    minAreaRect = []
    boxes = []

    for i,c in enumerate(cnts):
        # get the bounding rect
        x, y, w, h = cv2.boundingRect(c)
        
        # get the min area rect
        rect = cv2.minAreaRect(c)
        box = cv2.boxPoints(rect)

        # convert all coordinates floating point values to int
        box = np.int0(box)
        minAreaRect = box
        boxes.append((i,box,w,h))

        # draw a red 'nghien' rectangle
        cv2.drawContours(mask, [box], 0, (255, 0, 0), 5)

    #order all the contours by area from biggest to smallest
    boxes.sort(key=lambda x: x[2]*x[3], reverse=True)

    #now we want to select the second biggest box (since the biggest one is the contour of the whole image)
    minAreaRectangle = boxes[1][1]
    minAreaRectangleIndex = boxes[0][0]

    #show that this is the contour we want
    cv2.drawContours(mask, [minAreaRectangle], 0, (255, 255, 255), 3)

    #get x and y points and width/height of the countour we selected
    x,y,w,h = cv2.boundingRect(cnts[minAreaRectangleIndex])
    
    if w > h:
        temp = w
        w = h
        h = temp

    return [w, h]
