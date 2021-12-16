# -*- coding: utf-8 -*-
"""
This is the script used to train an activity recognition 
classifier on accelerometer data.

"""

import os
import sys
import numpy as np
from sklearn import model_selection
from sklearn.metrics import confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz
from features import extract_features
from util import slidingWindow, reorient, reset_vars
import pickle

import tqdm

window_size = 4
step_size = 4

X = [] ; Y = []
for data_file in ['data/tv-data.csv', 'data/indoor-data.csv', 'data/touchface-data.csv', 'data/touch2-data.csv', 'data/notouch-data.csv']:
    print(f"Reading data {data_file}...")
    data = np.genfromtxt(data_file, delimiter=',', skip_header=1)
    # remove nan columns
    data = data[:, ~np.isnan(data).any(axis=0)]
    for _, window in tqdm.tqdm(slidingWindow(data, window_size, step_size), total=((len(data)-window_size)//step_size)+1):
        feature_names, x = extract_features(window[:,:-1])
        X.append(x)
        Y.append(window[0][-1])
        
print(f"Loaded {len(X)} raw labelled activity data samples.")

# %%---------------------------------------------------------------------------
#
#                                Extract Features & Labels
#
# -----------------------------------------------------------------------------

# TODO: list the class labels that you collected data for in the order of label_index (defined in collect-labelled-data.py)
class_names = ["not touching", "touching face"]
   
X = np.asarray(X)
Y = np.asarray(Y)
n_features = len(X)

print("Finished feature extraction over {} windows".format(len(X)))
print("Unique labels found: {}".format(set(Y)))
print("\n")
sys.stdout.flush()

# %%---------------------------------------------------------------------------
#
#                                Train & Evaluate Classifier
#
# -----------------------------------------------------------------------------



"""
TODO: iterating over each fold, fit a decision tree classifier on the training set.
Then predict the class labels for the test set and compute the confusion matrix
using predicted labels and ground truth values. Print the accuracy, precision and recall
for each fold.
"""

cv = model_selection.KFold(n_splits=10, random_state=None, shuffle=True)
tree = DecisionTreeClassifier(criterion='entropy', max_depth=3)

# split data into train and test datasets using 10-fold cross validation
accuracy_list = []
precision_list = []
recall_list = []
for fold_idx, (train_index, test_index) in enumerate(cv.split(X)):
    X_train, X_test = X[train_index], X[test_index]
    Y_train, Y_test = Y[train_index], Y[test_index]
    tree.fit(X_train, Y_train)
    Y_pred = tree.predict(X_test)

    # calculate and print the average accuracy, precision and recall values over all 10 folds
    conf = confusion_matrix(Y_test, Y_pred)
    true_pos = np.diag(conf)
    false_pos = np.sum(conf, axis=0) - true_pos
    false_neg = np.sum(conf, axis=1) - true_pos
    accuracy = np.sum(true_pos) / np.sum(conf)
    precision = np.average(true_pos / (true_pos + false_pos))
    recall = np.average(true_pos / (true_pos + false_neg))
    accuracy_list.append(accuracy)
    precision_list.append(precision)
    recall_list.append(recall)
    print("FOLD", fold_idx)
    print("Accuracy:  %05.2f%%" % (100 * accuracy))
    print("Precision: %05.2f%%" % (100 * precision))
    print("Recall:    %05.2f%%" % (100 * recall))

print("AVERAGE OVER ALL FOLDS")
print("Accuracy:  %05.2f%%" % (100 * np.average(accuracy_list)))
print("Precision: %05.2f%%" % (100 * np.average(precision_list)))
print("Recall:    %05.2f%%" % (100 * np.average(recall_list)))

# train the decision tree classifier on entire dataset
tree.fit(X, Y)

# Save the classifier to disk
with open('classifier.pickle', 'wb') as f:
    pickle.dump(tree, f)
