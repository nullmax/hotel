# -*- coding: utf-8 -*-
"""
Created on Sat Dec  1 19:24:16 2018

@author: MAXNU
"""


import numpy as np

from sklearn import preprocessing, tree
from sklearn.ensemble import RandomForestClassifier

def trainAndClassify(clf, x_train, y_train, x_test, y_test, data):
    clf.fit(x_train, y_train)
    score = clf.score(x_test, y_test)
    print(score)
    return clf.predict(unknown_data)

fileName = 'data.npy'

data = np.load(fileName)

unknown_data=data[0:98]
unknown_data=unknown_data[:,0:8]

data = data[99:]
np.random.shuffle(data)
x = data[:,0:8]
y = data[:,9]

x = preprocessing.scale(x)
y = y.astype(np.int)

alpha = 0.75
n_train = int(len(data) * 0.7)

x_train = x[0:n_train]
y_train = y[0:n_train]

x_test = x[n_train+1:]
y_test = y[n_train+1:]

depth = 3
result1 = trainAndClassify(tree.DecisionTreeClassifier(max_depth=depth,random_state=0),
                           x_train, y_train,
                           x_test, y_test,
                           unknown_data)

result2 = trainAndClassify(RandomForestClassifier(n_estimators=64, max_depth=depth,random_state=0),
                           x_train, y_train,
                           x_test, y_test,
                           unknown_data)