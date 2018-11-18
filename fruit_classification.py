import cv2
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import OneHotEncoder

from dataframe_generator import *
from extraction_functions.extract_color import *
from extraction_functions.extract_dimensions import extract_dimensions


def train_k_neighbors_classifier():
    neigh = KNeighborsClassifier(n_neighbors=1)

    # df = pd.read_csv('./test.csv')
    df = pd.concat([pd.read_csv('./test.csv'), pd.read_csv('./train.csv')])

    n1 = int(len(df)*0.8)
    df = df.sample(frac=1)
    df1 = df[:n1]
    df2 = df[n1:]

    # df1 = pd.read_csv('./train.csv')
    # df2 = pd.read_csv('./fruits.csv')


    y1 = df1['fruit_type']
    y2 = df2['fruit_type']

    df1 = df1.drop(columns=['fruit_type', 'rgb()', 'name', 'width', 'height'])
    df2 = df2.drop(columns=['fruit_type', 'rgb()', 'name', 'width', 'height'])

    x1 = df1.values[:,1:]
    x2 = df2.values[:,1:]

    x1 = df1.values
    x2 = df2.values

    neigh.fit(x1, y1) 
    neigh.kneighbors_graph()
    y2p = neigh.predict(x2)

    results = pd.DataFrame({'y2': y2p, 'y': y2})
    results['score'] = results['y2']==results['y']
    print(results)
    print(sum(results['score'])/len(results))


def main():
    # generate_train_dataset()
    # generate_test_dataset()
    train_k_neighbors_classifier()

main()