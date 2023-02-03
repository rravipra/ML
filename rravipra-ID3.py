##############
# Name: Rthvik Raviprakash
# email: rravipra@purdue.edu
# Date: 10/17/2020

import numpy as np
import pandas as pd

def entropy(freqs):
    """
    entropy(p) = -SUM (Pi * log(Pi))
    ">>> entropy([10.,10.])
    1.0
    ">>> entropy([10.,0.])
    0
    ">>> entropy([9.,3.])
    0.811278
    """
    all_freq = sum(freqs)
    entropy = 0
    if not all_freq == 0:
        for fq in freqs:
            prob = fq * 1.0 / all_freq
            if abs(prob) > 1e-8:
                entropy += - ((fq * 1.0) / all_freq) * np.log2((fq * 1.0) / all_freq)
        return entropy
    else:
        return 0

def infor_gain(before_split_freqs, after_split_freqs):
    """
    gain(D, A) = entropy(D) - SUM ( |Di| / |D| * entropy(Di) )
    ">>> infor_gain([9,5], [[2,2],[4,2],[3,1]])
    0.02922
    """
    gain = entropy(before_split_freqs)
    overall_size = sum(before_split_freqs)
    if not overall_size == 0:
        for freq in after_split_freqs:
            ratio = sum(freq) * 1.0 / overall_size
            gain -= ratio * entropy(freq)
        return gain
    else:
        return 0

class Node(object):
    def __init__(self, l, r, attr, thresh, leaf):
        self.left_subtree = l
        self.right_subtree = r
        self.attribute = attr
        self.threshold = thresh
        self.leaf = leaf

class Tree(object):
    def __init__(self, ____):
        pass


def ID3(train_data, train_labels):
    # 1. use a for loop to calculate the infor-gain of every attribute

    data = pd.concat([train_data, train_labels], axis=1)

    if len(train_data.columns) == 0 or ((train_labels['survived'] == 0).all() == True) or ((train_labels['survived'] == 1).all() == True):
        if (train_labels['survived'] == 0).all():
            val = Node(None, None, None, 0, "leaf")
            return val
        elif (train_labels['survived'] == 1).all():
            val = Node(None, None, None, 1, "leaf")
            return val
        else:
            if (train_labels['survived'] == 1).sum() < (train_labels['survived'] == 0).sum():
                val = Node(None, None, None, 0, "leaf")
                return val
            else:
                val = Node(None, None, None, 1, "leaf")
                return val

    max_gain = 0
    the_chosen_attribute = ""
    the_chosen_threshold = 0

    count0 = 0
    count1 = 0

    for val in train_labels['survived']:
        if val == 0:
            count0 += 1
        if val == 1:
            count1 += 1

    for att in train_data:
        arr = train_data[att].unique()
        arr.sort()
        gain = 0
        threshold = 0
        for i in range(len(arr) - 1):
            avg = (arr[i] + arr[i + 1]) / 2
            count00 = 0
            count01 = 0
            count10 = 0
            count11 = 0
            val = np.array(train_data[att].values)
            for p in range(len(val)):
                if val[p] <= avg:
                    if data._get_value(p, len(data.columns) - 1, takeable=True) == 0:
                        count00 = count00 + 1
                    if data._get_value(p, len(data.columns) - 1, takeable=True) == 1:
                        count01 = count01 + 1
                if val[p] > avg:
                    if data._get_value(p, len(data.columns) - 1, takeable=True) == 0:
                        count10 = count10 + 1
                    if data._get_value(p, len(data.columns) - 1, takeable=True) == 1:
                        count11 = count11 + 1
            if infor_gain([count0, count1], [[count00, count01], [count10, count11]]) > gain:
                gain = infor_gain([count0, count1], [[count00, count01], [count10, count11]])
                threshold = avg

        if gain > max_gain:
            the_chosen_attribute = att
            the_chosen_threshold = threshold
            max_gain = gain

        # 1.1 pick a threshold
        # 1.2 split the data using the threshold
        # 1.3 calculate the infor_gain
        # 2. pick the attribute that achieve the maximum infor-gain
        # 3. build a node to hold the data;

        # 4. split the data into two parts.
    if max_gain == 0:
        return Node(None, None, 'survived', data['survived'].mode().iat[0], "leaf")

    df1 = train_data.copy()
    df2 = data.copy()
    left_part_train_data = pd.DataFrame(df1[df1.loc[:, the_chosen_attribute] <= the_chosen_threshold])
    left_part_train_label = pd.DataFrame(df2[df2.loc[:, the_chosen_attribute] <= the_chosen_threshold][["survived"]])
    right_part_train_data = pd.DataFrame(df1[df1.loc[:, the_chosen_attribute] > the_chosen_threshold])
    right_part_train_label = pd.DataFrame(df2[df2.loc[:, the_chosen_attribute] > the_chosen_threshold][["survived"]])

    left_part_train_data.drop(the_chosen_attribute, axis=1, inplace=True)
    right_part_train_data.drop(the_chosen_attribute, axis=1, inplace=True)

    current_node = Node(None, None, the_chosen_attribute, the_chosen_threshold, "not-leaf")
        # 5. call ID3() for the left parts of the data
    left_subtree = ID3(left_part_train_data, left_part_train_label)
        # 6. call ID3() for the right parts of the data.
    current_node.left_subtree = left_subtree

    right_subtree = ID3(right_part_train_data, right_part_train_label)

    current_node.right_subtree = right_subtree

    return current_node

def ID3_min_split(train_data, train_labels, min_split):
    data = pd.concat([train_data, train_labels], axis=1)

    if len(train_data) <= min_split or len(train_data.columns) == 0 or ((train_labels['survived'] == 1).all() == True) or ((train_labels['survived'] == 0).all() == True):
        if (train_labels['survived'] == 0).all():
            val = Node(None, None, None, 0, "leaf")
            return val
        elif (train_labels['survived'] == 1).all():
            val = Node(None, None, None, 1, "leaf")
            return val
        else:
            if (train_labels['survived'] == 1).sum() < (train_labels['survived'] == 0).sum():
                val = Node(None, None, None, 0, "leaf")
                return val
            else:
                val = Node(None, None, None, 1, "leaf")
                return val

    max_gain = 0
    the_chosen_attribute = ""
    the_chosen_threshold = 0

    count0 = 0
    count1 = 0

    for val in train_labels['survived']:
        if val == 0:
            count0 += 1
        if val == 1:
            count1 += 1

    for att in train_data:
        arr = train_data[att].unique()
        arr.sort()
        gain = 0
        threshold = 0
        for i in range(len(arr) - 1):
            avg = (arr[i] + arr[i + 1]) / 2
            count00 = 0
            count01 = 0
            count10 = 0
            count11 = 0
            val = np.array(train_data[att].values)
            for p in range(len(val)):
                if val[p] <= avg:
                    if data._get_value(p, len(data.columns) - 1, takeable=True) == 0:
                        count00 = count00 + 1
                    if data._get_value(p, len(data.columns) - 1, takeable=True) == 1:
                        count01 = count01 + 1
                if val[p] > avg:
                    if data._get_value(p, len(data.columns) - 1, takeable=True) == 0:
                        count10 = count10 + 1
                    if data._get_value(p, len(data.columns) - 1, takeable=True) == 1:
                        count11 = count11 + 1
            if infor_gain([count0, count1], [[count00, count01], [count10, count11]]) > gain:
                gain = infor_gain([count0, count1], [[count00, count01], [count10, count11]])
                threshold = avg

        if gain > max_gain:
            the_chosen_attribute = att
            the_chosen_threshold = threshold
            max_gain = gain

        # 1.1 pick a threshold
        # 1.2 split the data using the threshold
        # 1.3 calculate the infor_gain
        # 2. pick the attribute that achieve the maximum infor-gain
        # 3. build a node to hold the data;

        # 4. split the data into two parts.
    if max_gain == 0:
        return Node(None, None, 'survived', data['survived'].mode().iat[0], "leaf")

    df1 = train_data.copy()
    df2 = data.copy()
    left_part_train_data = pd.DataFrame(df1[df1.loc[:, the_chosen_attribute] <= the_chosen_threshold])
    left_part_train_label = pd.DataFrame(df2[df2.loc[:, the_chosen_attribute] <= the_chosen_threshold][["survived"]])
    right_part_train_data = pd.DataFrame(df1[df1.loc[:, the_chosen_attribute] > the_chosen_threshold])
    right_part_train_label = pd.DataFrame(df2[df2.loc[:, the_chosen_attribute] > the_chosen_threshold][["survived"]])

    left_part_train_data.drop(the_chosen_attribute, axis=1, inplace=True)
    right_part_train_data.drop(the_chosen_attribute, axis=1, inplace=True)

    current_node = Node(None, None, the_chosen_attribute, the_chosen_threshold, "not-leaf")
        # 5. call ID3() for the left parts of the data
    left_subtree = ID3_min_split(left_part_train_data, left_part_train_label, min_split)
        # 6. call ID3() for the right parts of the data.
    current_node.left_subtree = left_subtree

    right_subtree = ID3_min_split(right_part_train_data, right_part_train_label, min_split)

    current_node.right_subtree = right_subtree

    return current_node

def ID3_max_depth(train_data, train_labels, max_depth):
    data = pd.concat([train_data, train_labels], axis=1)

    if max_depth == 1 or len(train_data.columns) == 0 or ((train_labels['survived'] == 1).all() == True) or ((train_labels['survived'] == 0).all() == True):
        if (train_labels['survived'] == 0).all():
            val = Node(None, None, None, 0, "leaf")
            return val
        elif (train_labels['survived'] == 1).all():
            val = Node(None, None, None, 1, "leaf")
            return val
        else:
            if (train_labels['survived'] == 1).sum() < (train_labels['survived'] == 0).sum():
                val = Node(None, None, None, 0, "leaf")
                return val
            else:
                val = Node(None, None, None, 1, "leaf")
                return val

    max_gain = 0
    the_chosen_attribute = ""
    the_chosen_threshold = 0

    count0 = 0
    count1 = 0

    for val in train_labels['survived']:
        if val == 0:
            count0 += 1
        if val == 1:
            count1 += 1

    for att in train_data:
        arr = train_data[att].unique()
        arr.sort()
        gain = 0
        threshold = 0
        for i in range(len(arr) - 1):
            avg = (arr[i] + arr[i + 1]) / 2
            count00 = 0
            count01 = 0
            count10 = 0
            count11 = 0
            val = np.array(train_data[att].values)
            for p in range(len(val)):
                if val[p] <= avg:
                    if data._get_value(p, len(data.columns) - 1, takeable=True) == 0:
                        count00 = count00 + 1
                    if data._get_value(p, len(data.columns) - 1, takeable=True) == 1:
                        count01 = count01 + 1
                if val[p] > avg:
                    if data._get_value(p, len(data.columns) - 1, takeable=True) == 0:
                        count10 = count10 + 1
                    if data._get_value(p, len(data.columns) - 1, takeable=True) == 1:
                        count11 = count11 + 1
            if infor_gain([count0, count1], [[count00, count01], [count10, count11]]) > gain:
                gain = infor_gain([count0, count1], [[count00, count01], [count10, count11]])
                threshold = avg

        if gain > max_gain:
            the_chosen_attribute = att
            the_chosen_threshold = threshold
            max_gain = gain

        # 1.1 pick a threshold
        # 1.2 split the data using the threshold
        # 1.3 calculate the infor_gain
        # 2. pick the attribute that achieve the maximum infor-gain
        # 3. build a node to hold the data;

        # 4. split the data into two parts.
    if max_gain == 0:
        return Node(None, None, 'survived', data['survived'].mode().iat[0], "leaf")

    df1 = train_data.copy()
    df2 = data.copy()
    left_part_train_data = pd.DataFrame(df1[df1.loc[:, the_chosen_attribute] <= the_chosen_threshold])
    left_part_train_label = pd.DataFrame(df2[df2.loc[:, the_chosen_attribute] <= the_chosen_threshold][["survived"]])
    right_part_train_data = pd.DataFrame(df1[df1.loc[:, the_chosen_attribute] > the_chosen_threshold])
    right_part_train_label = pd.DataFrame(df2[df2.loc[:, the_chosen_attribute] > the_chosen_threshold][["survived"]])

    left_part_train_data.drop(the_chosen_attribute, axis=1, inplace=True)
    right_part_train_data.drop(the_chosen_attribute, axis=1, inplace=True)

    current_node = Node(None, None, the_chosen_attribute, the_chosen_threshold, "not-leaf")
        # 5. call ID3() for the left parts of the data
    left_subtree = ID3_max_depth(left_part_train_data, left_part_train_label, max_depth-1)
        # 6. call ID3() for the right parts of the data.
    current_node.left_subtree = left_subtree

    right_subtree = ID3_max_depth(right_part_train_data, right_part_train_label, max_depth-1)

    current_node.right_subtree = right_subtree

    return current_node


def k_fold(train_data, train_label, model, k,  min_split = None, max_depth = None):
    training_data = pd.concat([train_data, train_label], axis = 1)
    for t in range(k):
        df = np.array_split(training_data, k)
        val_data = df[t]
        del df[t]
        data = pd.concat(df)
        data = data.reset_index(drop=True)
        val_data = val_data.reset_index(drop = True)

        if model == "vanilla":
            tree = ID3(train_data, train_label)
        if model == "depth":
            tree = ID3_max_depth(train_data, train_label, max_depth)
        if model == "minSplit":
            tree = ID3_min_split(train_data, train_label, min_split)

        train_pred = prediction(data, tree)
        validation_pred = prediction(val_data, tree)

        train_acc = accuracy(train_pred)
        val_acc = accuracy(validation_pred)

        print("fold=" + str(t+1) + ", train set accuracy= " + str(train_acc) + ", validation set accuracy= " + str(val_acc))

def prediction(frame, tree):
    list_predictions_threshold = []

    #print(frame)
    #print(frame.loc[0, 'Sex'])
    for i in range(len(frame.index)):
        current = tree
        #print(i)
        #print(current.attribute)
        #print(current.leaf)
        #print(current.threshold)
        while current.leaf == "not-leaf":
            if frame.loc[i, current.attribute] <= current.threshold:
                current = current.left_subtree
            else:
                current = current.right_subtree
        list_predictions_threshold.append([frame.loc[i, 'survived'], current.threshold])

    return list_predictions_threshold

def accuracy(pred):
    count = 0
    for i in range(len(pred)):
        if pred[i][0] == pred[i][1]:
            count += 1

    return count / len(pred)

def test_accuracy(test_data, test_label, tree):

    data_frame = pd.concat([test_data, test_label], axis = 1)
    test_pred = prediction(data_frame, tree)
    test_acc = accuracy(test_pred)

    print("Test set accuracy= " + str(test_acc))

def size(tree):
    if tree == None:
        return 0
    else:
        return 1 + size(tree.left_subtree) + size(tree.right_subtree)

"""class PCA(object):
    def __init__(self, n_component):
        self.n_component = n_component

    def fit_transform(self, train_data):

    # [TODO] Fit the model with train_data and
    # apply the dimensionality reduction on train_data.

    def transform(self, test_data):

# [TODO] Apply dimensionality reduction to test_data."""

if __name__ == "__main__":
    # parse arguments
    import argparse

    parser = argparse.ArgumentParser(description='CS373 Homework2 Decision Tree')
    parser.add_argument('--trainFolder')
    parser.add_argument('--testFolder')
    parser.add_argument('--model')
    parser.add_argument('--crossValidK', type=int, default=5)
    args = parser.parse_args()
    print(args)

    if args.model == "vanilla":

        X_train = pd.read_csv(args.trainFolder + '/train-file.data', delimiter=',',index_col=None, engine='python')
        Y_train = pd.read_csv(args.trainFolder + '/train-file.label', delimiter=',',index_col=None, engine='python')

        X_test = pd.read_csv(args.testFolder + '/test-file.data', delimiter=',', index_col=None, engine='python')
        Y_test = pd.read_csv(args.testFolder + '/test-file.label', delimiter=',', index_col=None, engine='python')

        for column in X_train.columns:
            X_train[column].fillna(X_train[column].mode()[0], inplace=True)
        for col in X_test.columns:
            X_test[col].fillna(X_test[col].mode()[0], inplace=True)

        tree_root = ID3(X_train, Y_train)
        k_fold(X_train, Y_train, "vanilla", args.crossValidK)
        test_accuracy(X_test, Y_test, tree_root)

    if args.model == "depth":

        X_train = pd.read_csv(args.trainFolder + '/train-file.data', delimiter=',',index_col=None, engine='python')
        Y_train = pd.read_csv(args.trainFolder + '/train-file.label', delimiter=',',index_col=None, engine='python')

        X_test = pd.read_csv(args.testFolder + '/test-file.data', delimiter=',', index_col=None, engine='python')
        Y_test = pd.read_csv(args.testFolder + '/test-file.label', delimiter=',', index_col=None, engine='python')

        for column in X_train.columns:
            X_train[column].fillna(X_train[column].mode()[0], inplace=True)
        for col in X_test.columns:
            X_test[col].fillna(X_test[col].mode()[0], inplace=True)

        tree_root = ID3(X_train, Y_train, args.depth)
        k_fold(X_train, Y_train, "depth", args.crossValidK)
        test_accuracy(X_test, Y_test, tree_root)

    if args.model == "minSplit":

        X_train = pd.read_csv(args.trainFolder + '/train-file.data', delimiter=',',index_col=None, engine='python')
        Y_train = pd.read_csv(args.trainFolder + '/train-file.label', delimiter=',',index_col=None, engine='python')

        X_test = pd.read_csv(args.testFolder + '/test-file.data', delimiter=',', index_col=None, engine='python')
        Y_test = pd.read_csv(args.testFolder + '/test-file.label', delimiter=',', index_col=None, engine='python')

        for column in X_train.columns:
            X_train[column].fillna(X_train[column].mode()[0], inplace=True)
        for col in X_test.columns:
            X_test[col].fillna(X_test[col].mode()[0], inplace=True)

        tree_root = ID3(X_train, Y_train)
        k_fold(X_train, Y_train, args.model, args.crossValidK)
        test_accuracy(X_test, Y_test, tree_root)

    # build decision tree

    # predict on testing set & evaluate the testing accuracy



"""def PreOrder(root):
    if root:
        print(root.attribute)
        PreOrder(root.left_subtree)
        PreOrder(root.right_subtree)

X = pd.read_csv('C:\\Users\\user\\Downloads\\CS37300 HW2 (1)\\CS37300 HW2\\titanic-train.data', delimiter=',',
                    index_col=None, engine='python')
Y = pd.read_csv('C:\\Users\\user\\Downloads\\CS37300 HW2 (1)\\CS37300 HW2\\titanic-train.label', delimiter=',',
                    index_col=None, engine='python')

for column in X.columns:
    X[column].fillna(X[column].mode()[0], inplace=True)
print(k_fold(X,Y,"minSplit", 5, 2, 10))"""
