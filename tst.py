from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
from random import sample
import math


def hold_out(data):
    test_set_size = math.floor(len(data.index) / 3)
    test_set_indexes = sample(range(0,len(data.index)), test_set_size)
    test_set = pd.DataFrame(data, index=test_set_indexes)
    train_set = data.drop(test_set_indexes)
    return train_set, test_set


def my_test():
    df = pd.DataFrame(
        {'col1': [1, 2, 3, 4, 5, 6, 7], 'col2': list('ebd?aba'), 'col3': list('?bcdbbr'), 'col4': list('eewcae?')})
    print(df)
    test_set_size = math.floor(len(df.index) / 3)
    test_set_indexes = sample(range(0, len(df.index)), test_set_size)
    test_set = pd.DataFrame(df, index=test_set_indexes)
    print("test_set_size: ", test_set_size)
    print("test_set_indexes: ", test_set_indexes)
    print("## Test Set \n", test_set, "\n##")
    train_set = df.drop(test_set_indexes)


def test():
    testing_set_data = (pd.DataFrame(data, columns=range(1, 22))).values.tolist()
    testing_set_result = (pd.DataFrame(data, columns=[0])).values.tolist()


def prog(data):
    training_set_data = (pd.DataFrame(data, columns=range(1, 22))).values.tolist()
    training_set_result = (pd.DataFrame(data, columns=[0])).values.tolist()
    # training_set_result reformat
    tsr_reformat = []
    for i in range(0, len(training_set_result)):
        tsr_reformat.append(training_set_result[i][0])
    print(tsr_reformat)
    print(training_set_data)
    print(training_set_result)
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
knn = prog(training_set)
#test()
#my_test()
# hold_out(data)
# print(neigh.predict_proba([[0,1]]))
