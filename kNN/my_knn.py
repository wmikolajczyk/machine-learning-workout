import math
import operator
from typing import Callable

import numpy as np
import pandas as pd


def kNN(
    X: pd.DataFrame,
    y: pd.Series,
    query: np.array,
    k: int,
    dist_fn: Callable,
    choice_fn: Callable,
) -> float:
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
    columns = ["Sex", "Age", "Survived"]
    df = input_df.loc[:, columns]
    df["Sex"] = df["Sex"].astype("category").cat.codes
    # print category codes
    sex_cat_mapping = dict(enumerate(input_df["Sex"].astype("category").cat.categories))
    print(f"sex cat mapping: {sex_cat_mapping}")
    df = df.dropna()
    return df


if __name__ == "__main__":
    titanic_input_df = pd.read_csv("dataset/titanic.csv")
    df = preprocess_titanic(titanic_input_df)

    split_idx = 500

    train_df = df.loc[:split_idx]
    test_df = df.loc[split_idx:]

    target_colname = "Survived"
    X = train_df.drop(columns=[target_colname])
    y = train_df[target_colname]

    X_test = test_df.drop(columns=[target_colname])
    y_test = test_df[target_colname]

    result = kNN(X, y, X_test.iloc[5], k=3, dist_fn=euclidean_distance, choice_fn=mode)
    print(f"features: {X_test.columns}")
    print(f"test X: {X_test.iloc[5].values}, test y: {y_test.iloc[5]}, pred: {result}")
