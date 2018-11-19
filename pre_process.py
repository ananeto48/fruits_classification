import cv2
import numpy as np
import pandas as pd
from os import listdir
from string import digits

from extraction_functions.extract_color import *
from extraction_functions.extract_dimensions import *

def add_white_border(img):
    shape=img.shape
    w=shape[1]
    h=shape[0]

    base_size=h+20,w+20,3
    #make a 3 channel image for base which is slightly larger than target img
    base=np.zeros(base_size,dtype=np.uint8)
    cv2.rectangle(base,(0,0),(w+20,h+20),(255,255,255),30)#really thick white rectangle
    base[10:h+10,10:w+10]=img #this works
    return base


def generate_train_dataset():
    i = 0
    data = [[0 for x in range(9)] for y in range(5000)]
    print("Generating train dataframe...")

    for directory in listdir('./kaggle_images'):
        if directory == '.DS_Store':
            continue
        
        for img in listdir('./kaggle_images/' + directory):
            img_matrix = cv2.imread(str('./kaggle_images/' + directory + '/' + img), 1)
            img_matrix = cv2.cvtColor(img_matrix, cv2.COLOR_BGR2RGB)
            img_matrix = add_white_border(img_matrix)

            print("Extracting color...")
            r, g, b = extract_color_threshold(img_matrix)

            print("Extracting dimensions...")
            w, h = extract_dimensions_threshold(img_matrix)

            #get fruit type from directory name, rgb format from extract_color and width_heigth_proportion from width and height
            rgb = ("rgb(%d,%d,%d)" % (r, g, b))
            width_heigth_proportion = w/h

            #insert data into data matrix
            return [img, directory, r, g, b, rgb, w, h, width_heigth_proportion]

            #insert data into data matrix
            data[i] = [name, directory, r, g, b, rgb, w, h, width_heigth_proportion]

            i = i + 1
            print('-'*60)

    #load data to dataframe
    train_df = pd.DataFrame(
        data=data, 
        index=range(0, 5000), 
        columns=['name', 'fruit_type', 'r', 'g', 'b', 'rgb()', 'width', 'height', 'width_heigth_proportion']
    )

    #filter dataframe to eliminate extra rows
    train_df = train_df[train_df['r'] != 0]
    sorted_train_df = train_df.sort_values(['r'])
    sorted_train_df.to_csv('train.csv')
    print(sorted_train_df)


def generate_test_dataset():
    i = 0
    data = [[0 for x in range(9)] for y in range(200)]

    for img in listdir('./photos_images'):
        print("Generating test dataframe...")
        print(img)
        img_matrix = cv2.imread(str('./photos_images/' + img), 1)
        
        #extract image params
        print("Extracting color...")
        r, g, b = extract_color_threshold(img_matrix)

        print("Extracting dimensions...")
        w, h = extract_dimensions_threshold(img_matrix)

        #get fruit type from directory name, rgb format from extract_color and width_heigth_proportion from width and height
        rgb = ("rgb(%d,%d,%d)" % (r, g, b))
        width_heigth_proportion = w/h

        #get fruit type from image name, rgb format from extract_color and width_heigth_proportion from width and height
        fruit_type = img[:-4].translate({ord(k): None for k in digits}).capitalize()

        #insert data into data matrix
        data[i] =  [img, fruit_type, r, g, b, rgb, w, h, width_heigth_proportion]

        i = i + 1
        print('-'*60)


    #load data to dataframe
    test_df = pd.DataFrame(
        data=data, 
        index=range(0,200), 
        columns=['name', 'fruit_type', 'r', 'g', 'b', 'rgb()', 'width', 'height', 'width_heigth_proportion']
    )

    #filter dataframe to eliminate extra rows
    test_df = test_df[test_df['r'] != 0]
    sorted_test_df = test_df.sort_values(['width_heigth_proportion'])
    sorted_test_df.to_csv('test.csv')
    print(sorted_test_df)

def pre_process_image(img):
    print("Processing image...")
    img_matrix = cv2.imread(img, 1)
    
    #extract image params
    print("Extracting color...")
    r, g, b = extract_color_threshold(img_matrix)


    print("Extracting dimensions...")
    w, h = extract_dimensions_threshold(img_matrix)

    #get fruit type from directory name, rgb format from extract_color and width_heigth_proportion from width and height
    rgb = ("rgb(%d,%d,%d)" % (r, g, b))
    width_heigth_proportion = w/h

    #insert data into data matrix
    return [r, g, b, width_heigth_proportion]
