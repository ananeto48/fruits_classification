import cv2
import numpy as np
from matplotlib import pyplot as plt

def extract_color(img):
    #converting image to graysclae to find threshold
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    gray = cv2.bilateralFilter(gray, 11, 17, 17)
    ret, thresh = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

    #finding image edges and enhancing them
    edged = cv2.Canny(gray, 50, 100)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    dilated = cv2.dilate(edged, kernel)

    #get edged image contours
    _, cnts, _ = cv2.findContours(dilated.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    #create a mask (black image) and draw contours of the fruit on the mask
    mask = np.zeros_like(img)

    #insert fruit area of original image onto a new image with black background
    out = np.zeros_like(img)
    out[mask==255] = img[mask==255]
    cv2.fillPoly(mask, pts =cnts, color=(255,255,255))
    
    #put back all 3 color dimensions on the mask
    img2 = img.copy()
    mask[:,:,1] = mask[:,:,0]
    mask[:,:,2] = mask[:,:,0]
    img2[mask!=255] = 0

    #get all the rgb values for the mask
    r = img2[:,:,0]
    g = img2[:,:,1]
    b = img2[:,:,2]

    #filter all the rgb values for the mask to eliminate the ones that are 0 (black)
    r = r[r!=0]
    g = g[g!=0]
    b = b[b!=0]

    #find color by calculating the mean of each color component
    return [int(r.mean()), int(g.mean()), int(b.mean())]

def extract_test_color(img):
    #converting image to graysclae to find threshold
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 11, 17, 17)
    ret, thresh = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

    #finding image edges and enhancing them
    edged = cv2.Canny(gray, 50, 100)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    dilated = cv2.dilate(edged, kernel)

    #get edged image contours
    _, cnts, _ = cv2.findContours(dilated.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    #create a mask (black image) and draw contours of the fruit on the mask
    mask = np.zeros_like(img)

    #insert fruit area of original image onto a new image with black background
    out = np.zeros_like(img)
    out[mask==255] = img[mask==255]
    cv2.fillPoly(mask, pts =cnts, color=(255,255,255))
    
    #put back all 3 color dimensions on the mask
    img2 = img.copy()
    mask[:,:,1] = mask[:,:,0]
    mask[:,:,2] = mask[:,:,0]
    img2[mask!=255] = 0
    plt.imshow(img2)

    #get all the rgb values for the mask
    r = img2[:,:,0]
    g = img2[:,:,1]
    b = img2[:,:,2]

    #filter all the rgb values for the mask to eliminate the ones that are 0 (black)
    r = r[r!=0]
    g = g[g!=0]
    b = b[b!=0]

    #find color by calculating the mean of each color component
    return [int(r.mean()), int(g.mean()), int(b.mean())]

def extract_train_color(img):
    #converting image to graysclae to find threshold
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 11, 17, 17)
    ret, thresh = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

    #finding image edges and enhancing them
    edged = cv2.Canny(gray, 50, 100)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    dilated = cv2.dilate(edged, kernel)

    #get edged image contours
    _, cnts, _ = cv2.findContours(dilated.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    #create a mask (black image) and draw contours of the fruit on the mask
    mask = np.zeros_like(img)

    #insert fruit area of original image onto a new image with black background
    out = np.zeros_like(img)
    out[mask==255] = img[mask==255]
    cv2.fillPoly(mask, pts =cnts, color=(255,255,255))
    
    #put back all 3 color dimensions on the mask
    img2 = img.copy()
    mask[:,:,1] = mask[:,:,0]
    mask[:,:,2] = mask[:,:,0]
    img2[mask!=255] = 0

    #get all the rgb values for the mask
    r = img2[:,:,0]
    g = img2[:,:,1]
    b = img2[:,:,2]

    #filter all the rgb values for the mask to eliminate the ones that are 0 (black)
    r = r[r!=0]
    g = g[g!=0]
    b = b[b!=0]

    return [int(r.mean()), int(g.mean()), int(b.mean())]
