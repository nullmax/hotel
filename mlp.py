# -*- coding: utf-8 -*-
"""
Created on Sat Dec  1 10:49:01 2018

@author: MAXNU
"""

import numpy as np

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import SGD

from sklearn import preprocessing

fileName = 'data.npy'

data = np.load(fileName)

unknown_data=data[0:98]

data = data[99:]
np.random.shuffle(data)
x = data[:,0:9]
y = data[:,9]

x = preprocessing.scale(x)
y = keras.utils.to_categorical(y, num_classes=6)

alpha = 0.75
n_train = int(len(data) * 0.7)

x_train = x[0:n_train]
y_train = y[0:n_train]

x_test = x[n_train+1:]
y_test = y[n_train+1:]

model = Sequential()
model.add(Dense(128, activation='relu', input_dim=9))
model.add(Dropout(0.5))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(6, activation='softmax'))

sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy',
              optimizer=sgd,
              metrics=['accuracy'])

model.fit(x_train, y_train,
          epochs=32,
          batch_size=32,
          validation_split=0.25)
score = model.evaluate(x_test, y_test, batch_size=32)

for i in range(len(score)):
    print(model.metrics_names[i], ": ", score[i])
    
unknown_data=unknown_data[:,0:9]
result = model.predict_classes(unknown_data, batch_size=32)
