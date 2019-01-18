from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import *
import pandas as pd
from random import sample
import math


def hold_out(data):
    test_set_size = math.floor(len(data.index) / 3)
    test_set_indexes = sample(range(0,len(data.index)), test_set_size)
    test_set = pd.DataFrame(data, index=test_set_indexes)
    train_set = data.drop(test_set_indexes)
    return train_set, test_set


def test(test_data):
    testing_set_data = (pd.DataFrame(test_data, columns=range(1, len(test_data.columns)))).values.tolist()
    testing_set_result = (pd.DataFrame(test_data, columns=[0])).values.tolist()
    predicted = knn.predict(testing_set_data)
    precision, recall, fscore = (precision_recall_fscore_support(testing_set_result, predicted, beta=1, average='binary'))[0:3]
    print("Precesion= ", precision)
    print("Recall= ", recall)
    print("fscore(beta=1)= ", fscore)


def main(data):
    training_set_data = (pd.DataFrame(data, columns=range(1, len(data.columns)))).values.tolist()
    training_set_result = (pd.DataFrame(data, columns=[0])).values.tolist()
    # training_set_result reformat
    tsr_reformat = []
    for i in range(0, len(training_set_result)):
        tsr_reformat.append(training_set_result[i][0])
    neigh = KNeighborsClassifier(n_neighbors=3)
    neigh.fit(training_set_data, tsr_reformat)
    return neigh


# fill missing value with mode
def fill_missing(data):
    for col in data.columns:
        mod = data[col].mode()[0]
        data[col] = data[col].replace('?', mod)


# convert nominal to numeric data
def nominal_to_numeric(data):
    obj_cols = data.select_dtypes(['object']).columns
    data[obj_cols] = data[obj_cols].astype('category')
    cat_cols = data.select_dtypes(['category']).columns
    data[cat_cols] = data[cat_cols].apply(lambda x: x.cat.codes)


data = pd.read_csv('data/agaricus-lepiota.data', sep=",", header=None)
fill_missing(data)
nominal_to_numeric(data)
training_set, testing_set = hold_out(data)
knn = main(training_set)
test(testing_set)
