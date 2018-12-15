# -*- coding: utf-8 -*-
"""
Created on Sat Dec  1 19:24:16 2018

@author: MAXNU
"""

import numpy as np

import graphviz

from sklearn import preprocessing, tree
from sklearn.ensemble import RandomForestClassifier

features=['清洁程度','舒适程度','位置','设施/服务','员工素质','性价比','免费WiFi','评论数','价格']
labels=['非常好', '好', '很棒', '优异的', '好极了']

def trainAndClassify(clf, x_train, y_train, x_test, y_test, data, draw=False):
    clf.fit(x_train, y_train)
    score = clf.score(x_test, y_test)
    print(score)
    predict_data = clf.predict(unknown_data)
    if draw == True:
        dot_data = tree.export_graphviz(clf, out_file = None,
                                        feature_names=features,
                                        class_names=labels,
                                        filled=True, rounded=True,
                                        special_characters=True)
        graph = graphviz.Source(dot_data)
        graph.render('hotel')
        return predict_data, graph
    return predict_data

fileName = 'data.npy'

data = np.load(fileName)

unknown_data=data[0:98]
unknown_data=unknown_data[:,0:9]

data = data[99:]
np.random.shuffle(data)
x = data[:,0:9]
y = data[:,9]

y = y.astype(np.int)

alpha = 0.75
n_train = int(len(data) * 0.7)

x_train = x[0:n_train]
y_train = y[0:n_train]

x_test = x[n_train+1:]
y_test = y[n_train+1:]

depth = 3
result1, graph = trainAndClassify(tree.DecisionTreeClassifier(max_depth=depth, random_state=0),
                           x_train, y_train,
                           x_test, y_test,
                           unknown_data, True)
