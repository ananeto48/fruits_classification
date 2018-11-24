import cv2
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import OneHotEncoder
from os import listdir

from helpers.pre_process import *
from extraction_functions.extract_color import *
from extraction_functions.extract_dimensions import *


def train_k_neighbors_classifier():
    neigh = KNeighborsClassifier(n_neighbors=5)

    # df = pd.read_csv('./test.csv')
    df = pd.concat([pd.read_csv('./test.csv'), pd.read_csv('./train.csv')])

    n1 = int(len(df)*0.8)
    df = df.sample(frac=1)
    y = df['fruit_type']

    cols = [0, 1, 2, 6, 7, 8, 10]
    df = df.drop(df.columns[cols], axis=1)

    df1 = df[:n1]
    df2 = df[n1:]


    y1 = y[:n1]
    y2 = y[n1:]

    x1 = df1.values
    x2 = df2.values

    neigh.fit(x1, y1) 
    neigh.kneighbors_graph()
    y2p = neigh.predict(x2)

    results = pd.DataFrame({'y2': y2p, 'y': y2})
    results['score'] = results['y2']==results['y']

    score = int(sum(results['score'])/len(results) * 100)

    print('SCORE: ', score, '%')
    return neigh

def classify_fruit(model, img):
    x = np.asarray(pre_process_image(img))
    X = np.asmatrix(x)
   
    print(X)
    y = model.predict(X)
    print(y)

def main():
    # generate_train_dataset()
    generate_test_dataset()
    model = train_k_neighbors_classifier()

    for img in listdir('./images_to_classify'):
        if img == '.DS_Store':
            continue

        print(img)
        classify_fruit(model, './images_to_classify/' + img)
        print("-"*40)

main()