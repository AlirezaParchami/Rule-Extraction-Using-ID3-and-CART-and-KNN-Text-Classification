from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import numpy as np


def hold_out(data):
    print(data)
    X = [[0, 0], [1, 1], [2, 2], [3, 3]]


def test():
    df = pd.DataFrame(
        {'col1': [1, 2, 3, 4, 5, 6], 'col2': list('ebd?ab'), 'col3': list('?bcdbb'), 'col4': list('eecae?')})
    mod = df['col4'].mode()[0]
    print("mod: ", mod , "  type: " , type(mod))
    print(df)
    print("----------------")
    for col in df.columns:
        mod = df[col].mode()[0]
        df[col] = df[col].replace('?', mod)
        #for col in df.columns:
        #    a = col.strip().replace("a" , mod)
    print(df)


def prog(data):
    training_set = (pd.DataFrame(data, columns=range(1, 22))).values.tolist()
    training_set_result = (pd.DataFrame(data, columns=[0])).values.tolist()
    # training_set_result reformat
    tsr_reformat = []
    for i in range(0, len(training_set_result)):
        tsr_reformat.append(training_set_result[i][0])
    print(tsr_reformat)
    print(training_set)
    print(training_set_result)
    neigh = KNeighborsClassifier(n_neighbors=3)
    neigh.fit(training_set, tsr_reformat)
#   print
    sample = training_set[0:2]
    print("sample: ", sample)
    print("sample predict: ", neigh.predict(sample))


# fill missing value with modal
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
prog(data)
# test()
# hold_out(data)
# print(neigh.predict_proba([[0,1]]))
