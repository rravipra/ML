import pandas as pd
import numpy as np
from numpy import random
import statistics

#import matplotlib.pyplot as plt

X = pd.read_csv('C:\\Users\\user\\Downloads\\HW3-CS373-V2\\HW3-CS373\\titanic-train.data', delimiter=',',
                    index_col=None, engine='python')
Y = pd.read_csv('C:\\Users\\user\\Downloads\\HW3-CS373-V2\\HW3-CS373\\titanic-train.label', delimiter=',',
                    index_col=None, engine='python')

X_test = pd.read_csv('C:\\Users\\user\\Downloads\\HW3-CS373-V2\\HW3-CS373\\titanic-test.data', delimiter=',',
                    index_col=None, engine='python')
Y_test = pd.read_csv('C:\\Users\\user\\Downloads\\HW3-CS373-V2\\HW3-CS373\\titanic-test.label', delimiter=',',
                    index_col=None, engine='python')

for col in X_test.columns:
    X_test[col].fillna(X_test[col].mode()[0], inplace=True)

for column in X.columns:
    X[column].fillna(X[column].mode()[0], inplace=True)

def convert(val):
    if val == 0:
        return -1
    else:
        return 1

for i in range(len(Y.index)):
    Y['survived'][i] = convert(Y['survived'][i])

for j in range(len(Y_test.index)):
    Y_test['survived'][j] = convert(Y_test['survived'][j])

data1 = pd.concat([X,Y], axis=1)
data_test = pd.concat([X_test,Y_test], axis=1)
#print(data1)

def dot_product(arr1, arr2, l):
    dot = 0
    for i in range(l):
        dot = dot + (arr1[i] * arr2[i])

    return dot

def sign(val):
    if val >= 0:
        return 1
    else:
        return -1

def compare(arr1, arr2, l):
    k = 0
    for i in range(l):
        if arr1[i] == arr2[i]:
            k += 1

    if k == l:
        return True
    else:
        return False

def weight_bias_index(data):
    w = [0,0,0,0,0,0,0]
    b = 0
    i = 0
    classified = False
    attr = []

    labels = data['survived'].values
    max_acc = 0
    w_max = []
    b_max = 0

    while (not classified) and i < 100:
        j = 0
        for s in data.index:
            y = labels[j]
            j += 1
            x = X.iloc[s].values
            if (y * (dot_product(w, x, 7) + b)) <= 0:
                w = w + (y * x)
                b = b + y

        sign_arr = []

        #attr.append(accuracy(predictions(data1,w,b), data1['survived'].values))

        for s_1 in data.index:
            x_1 = X.iloc[s_1].values
            sign_val = sign(dot_product(w, x_1, 7) + b)
            sign_arr.append(sign_val)

        acc = train_accuracy(sign_arr, labels)

        if acc > max_acc:
            max_acc = acc
            w_max = w
            b_max = b

        #print(acc)
        #print(sign_arr)

        if compare(sign_arr, labels, len(sign_arr)):
        #if sign_arr == labels:
            classified = True
        i += 1

    return w_max, b_max, i, max_acc

def predictions(data, weight, bias):
    array = []
    for i in data.index:
        x_1 = X.iloc[i].values
        sign_val = sign(dot_product(weight, x_1, 7) + bias)
        array.append(sign_val)

    return array

def accuracy(arr1, arr2):
    k = 0
    for i in range(len(arr1)):
        if arr1[i] == arr2[i]:
            k += 1

    print('Test Accuracy=', k / len(arr1))

def train_accuracy(arr1, arr2):
    k = 0
    for i in range(len(arr1)):
        if arr1[i] == arr2[i]:
            k += 1

    return k/len(arr1)

def hinge_loss(pred, true):
    if pred*true >= 1:
        return 0
    else:
        return (1 - (pred*true))

def avg_hinge_loss(arr1, arr2):
    l = len(arr1)
    sum = 0

    for i in range(l):
        sum = sum + hinge_loss(arr1[i], arr2[i])

    print('Hinge LOSS=', sum/l)
    return sum/l

#print(weight_bias_index(data))
#print(predictions(data1, weight_bias_index(data1)[0], weight_bias_index(data1)[1]))
#print(data['survived'].values)
#print(weight_bias_index(data1))
#print(data_test.iloc[0].values)
#print(predictions(data_test, [0,1,4,5,6,7,8], 7))
#print(weight_bias_index(data1)[0])

#print(predictions(data_test, weight_bias_index(data1)[0], weight_bias_index(data1)[1]))
#print(weight_bias_index(data1))

#print('acc')
#print(accuracy(predictions(data_test, weight_bias_index(data1)[0], weight_bias_index(data1)[1]), data_test['survived'].values))
#print(data_test['survived'].values)

#print(dot_product([1,2,3,4], [1,1,1,1], 4))
#print(weight_bias_index(data1))

#print(avg_hinge_loss(predictions(data_test, weight_bias_index(data1)[0], weight_bias_index(data1)[1]), data_test['survived'].values))
#print(data1['survived'].values)

def cross_validation(data, perc):
    sum = 0

    for i in range(10):
        train = data.sample(frac = perc)
        test = data.drop(train.index)
        arr1 = predictions(test, weight_bias_index(train)[0], weight_bias_index(train)[1])
        arr2 = test['survived'].values
        sum = sum + avg_hinge_loss(arr1, arr2)

    return sum/10

#avg_hinge_loss(predictions(data_test, weight_bias_index(data1)[0], weight_bias_index(data1)[1]), data_test['survived'].values)
#accuracy(predictions(data_test, weight_bias_index(data1)[0], weight_bias_index(data1)[1]), data_test['survived'].values)
"""for i in [0.01, 0.10, 0.50]:
    print("Mean Hinge Loss for " + str(i) + " : ", cross_validation(data1, i))"""

arr = Y_test['survived'].values
val = statistics.mode(arr)
arr1 = []

for i in range(len(arr)):
    arr1.append(val)

avg_hinge_loss(arr1, data_test['survived'].values)

def error(arr1, arr2):
    k = 0
    for i in range(len(arr1)):
        if arr1[i] == arr2[i]:
            k += 1

    print("baseline default error = ", k/len(arr1))

print(error(arr1, data_test['survived'].values))
if __name__ == "__main__":
    # parse arguments
    import argparse
    import sys

    """parser = argparse.ArgumentParser(description='CS373 Homework3 NBC')
    parser.add_argument('--trainFileData')
    parser.add_argument('--trainFileLabel')
    parser.add_argument('--testFileData')
    parser.add_argument('--testFileLabel')
    args = parser.parse_args()
    print(args)"""

    X_train = pd.read_csv(sys.argv[1], delimiter=',',index_col=None, engine='python')
    Y_train = pd.read_csv(sys.argv[2], delimiter=',',index_col=None, engine='python')

    X_test = pd.read_csv(sys.argv[3], delimiter=',', index_col=None, engine='python')
    Y_test = pd.read_csv(sys.argv[4], delimiter=',', index_col=None, engine='python')

    for column in X_train.columns:
        X_train[column].fillna(X_train[column].mode()[0], inplace=True)
    for col in X_test.columns:
        X_test[col].fillna(X_test[col].mode()[0], inplace=True)

    for i in range(len(Y.index)):
        Y_train['survived'][i] = convert(Y_train['survived'][i])

    for j in range(len(Y_test.index)):
        Y_test['survived'][j] = convert(Y_test['survived'][j])

    data = pd.concat([X_train, Y_train], axis=1)
    data_test = pd.concat([X_test, Y_test], axis=1)

    val = X_test['survived'].mode()[0]
    arr = data_test['survived'].values
    arr1 = []

    for i in range(len(arr)):
        arr1.append(val)

    avg_hinge_loss(arr1, data_test['survived'].values)
    accuracy(predictions(data_test, weight_bias_index(data)[0], weight_bias_index(data)[1]), data_test['survived'].values)