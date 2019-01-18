from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import numpy as np


def hold_out():
    data = pd.read_csv('data/agaricus-lepiota.data', sep=",", header=None)
    obj_cols = data.select_dtypes(['object']).columns
    data[obj_cols] = data[obj_cols].astype('category')
    print(data.dtypes)
    cat_cols = data.select_dtypes(['category']).columns
    data[cat_cols] = data[cat_cols].apply(lambda x: x.cat.codes)
    print(data)
    # print(pd['1']['2'])
    X = [[0, 0], [1, 1], [2, 2], [3, 3]]
    # print(X)
    # print(data.values.tolist())


def test():
    df = pd.DataFrame(
        {'col1': [1, 2, 3, 4, 5, 6], 'col2': list('ebdcab'), 'col3': list('abcdbb'), 'col4': list('eecaae')})
    tmp = df.select_dtypes(['object']).columns
    df[tmp] = df[tmp].astype('category')
    cat_columns = df.select_dtypes(['category']).columns
    df[cat_columns] = df[cat_columns].apply(lambda x: x.cat.codes)
    print(df)
    print("----------------")
    # for col in df.columns:
    #    print(df[col].mode())
    mod = df['col4'].mode()[0]
    print("mode: ", mod)
    print(df[df.col4 == 0]['col4'])
    a = df['col4'].replace(['0'], mod)
    # df.replace(df[df.col4==0]['col4'] , mod)
    print("-----------------")
    print(a)


def prog(data):
    training_set = (pd.DataFrame(data, columns=range(1, 22))).values.tolist()
    training_set_result = (pd.DataFrame(data, columns=[0])).values.tolist()
    tsr_reformat = []
    for i in range(0, len(training_set_result)):
        tsr_reformat.append(training_set_result[i][0])
    print(tsr_reformat)
    print(training_set)
    print(training_set_result)
    neigh = KNeighborsClassifier(n_neighbors=3)
    #neigh.fit(X, z)
    #print(neigh.predict([[pd.DataFrame(data, index=[3], columns=range(1, 22)).values]]))


# X = [[0,0], [1,1], [2,2], [3,3]]
# y = [0, 0, 1, 1]
data = pd.read_csv('data/agaricus-lepiota.data', sep=",", header=None)
prog(data)
#test()
# hold_out()
# print(neigh.predict_proba([[0,1]]))
