import math
import operator
from typing import Callable

import pandas as pd
import numpy as np


def kNN(X: pd.DataFrame, y: pd.Series, query: np.array, k: int, dist_fn: Callable, choice_fn: Callable) -> float:
    ranking = []
    for index, row in X.iterrows():
        dist = dist_fn(row, query)
        ranking.append((index, dist))

    ranking_sorted = sorted(ranking, key=operator.itemgetter(1))
    selected_idx = [x[0] for x in ranking_sorted[:k]]
    selected_labels = y.loc[selected_idx]
    prediction = choice_fn(selected_labels)
    return prediction


def euclidean_distance(point1: pd.Series, point2: pd.Series):
    assert len(point1) == len(point2)
    dist = 0
    for p1_val, p2_val in zip(point1, point2):
        dist += math.pow(p1_val - p2_val, 2)
    dist = math.sqrt(dist)
    return dist


def mode(series):
    return float(series.mode()[0])


def return_similar(series):
    return series


def preprocess_titanic(input_df):
    columns = ['Sex', 'Age', 'Survived']
    df = input_df.loc[:, columns]
    df['Sex'] = df['Sex'].astype('category').cat.codes
    # print category codes
    sex_cat_mapping = dict(enumerate(input_df['Sex'].astype('category').cat.categories))
    print(f'sex cat mapping: {sex_cat_mapping}')
    df = df.dropna()
    return df


if __name__ == '__main__':
    # df = pd.DataFrame([
    #    [22, 1],
    #    [23, 1],
    #    [21, 1],
    #    [18, 1],
    #    [19, 1],
    #    [25, 0],
    #    [27, 0],
    #    [29, 0],
    #    [31, 0],
    #    [45, 0],
    # ], columns=['a', 'b'])
    # target_col = 'b'
    #
    # X = df.drop(columns=[target_col])
    # y = df[target_col]
    # query = pd.Series([33])
    #
    # k = 3
    # # regression
    # print(kNN(X, y, query, k=k, dist_fn=euclidean_distance, choice_fn=np.mean))
    # # classification
    # print(kNN(X, y, query, k=k, dist_fn=euclidean_distance, choice_fn=mode))

    # titanic_input_df = pd.read_csv('dataset/titanic.csv')
    # df = preprocess_titanic(titanic_input_df)
    #
    # split_idx = 500
    #
    # train_df = df.loc[:split_idx]
    # test_df = df.loc[split_idx:]
    #
    # target_colname = 'Survived'
    # X = train_df.drop(columns=[target_colname])
    # y = train_df[target_colname]
    #
    # X_test = test_df.drop(columns=[target_colname])
    # y_test = test_df[target_colname]
    #
    # result = kNN(X, y, X_test.iloc[5], k=3, dist_fn=euclidean_distance, choice_fn=mode)
    # print(result)

    movies_input_df = pd.read_csv('dataset/movies_recommendation_data.csv')

    target_colname = 'Movie Name'
    cols_to_drop = ['Movie ID']
    X = movies_input_df.drop(columns=cols_to_drop + [target_colname])
    y = movies_input_df[target_colname]

    query = pd.Series([7.2, 1, 1, 0, 0, 0, 0, 1, 0])
    result = kNN(X, y, query, k=5, dist_fn=euclidean_distance, choice_fn=return_similar)
    print(result)

