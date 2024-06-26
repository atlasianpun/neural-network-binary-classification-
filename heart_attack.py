# -*- coding: utf-8 -*-
"""Heart_attack.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1gvgW92YONxGvrXicOyTLJ0ksr8LkEv7l
"""

#dataset - https://www.kaggle.com/datasets/yasserh/wine-quality-dataset?resource=download
#the data is stored on the drive and hence I mount my drive
# we chose this dataset cause we are alcoholics
from google.colab import drive
drive.mount('/content/drive')

import pandas as pd
import math
import numpy as np
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler, RobustScaler, Normalizer, OneHotEncoder
import plotly
import plotly.express as px
plotly.offline.init_notebook_mode (connected = True)
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as st
import hyperopt
from hyperopt import hp, fmin, tpe, Trials, space_eval, STATUS_OK
import tensorflow as tf
from tensorflow import keras
from keras import layers,Sequential
from keras.models import Sequential
from keras.layers import Dense , Activation, Dropout
from keras.optimizers import Adam ,RMSprop
from keras import  backend as K
from keras.utils import to_categorical, plot_model

file_path = '/content/drive/MyDrive/ML/heart.csv' # punya

df = pd.read_csv(file_path)

df.info()

df.describe().T.style.background_gradient(subset=['std'], cmap='Reds').background_gradient(subset=['mean'], cmap='Greens')

"""
outliers = df.loc[df['volatile acidity'] > 1.4]
outliers = pd.concat([outliers, df.loc[df['citric acid'] > 0.9]], axis=0)
outliers = pd.concat([outliers, df.loc[df['chlorides'] > 0.5]], axis=0)
outliers = pd.concat([outliers, df.loc[df['free sulfur dioxide'] > 60]], axis=0)
outliers = pd.concat([outliers, df.loc[df['total sulfur dioxide'] > 200]], axis=0)
outliers = pd.concat([outliers, df.loc[df['sulphates'] > 1.75]], axis=0)
outliers = pd.concat([outliers, df.loc[df['alcohol'] > 14]], axis=0)

outliers = outliers[~outliers.duplicated()]
"""

#df = df.drop(outliers.index)

df

features = ['age','sex','cp','trtbps','chol','fbs','restecg','thalachh','exng','oldpeak','slp','caa','thall']
X = df.loc[:, features]
Y = df.loc[:,'output']

# running the scalar for the X values
"""
scaler = StandardScaler()
X[features] = scaler.fit_transform(X[features])
X[features] = scaler.transform(X[features])
"""
X = np.array(X)

# Y = one_hot_encoder.fit_transform(np.array(Y).reshape(-1, 1))
Y = np.array(Y)
num_labels = len(np.unique(Y))
Y= to_categorical(Y, num_labels)

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.3)
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.1)

print("Shape:", x_train.shape)
print("Data type:", x_train.dtype)
print("Number of dimensions:", x_train.ndim)
print("Size (total number of elements):", x_train.size)
print("Item size (size in bytes of each element):", x_train.itemsize)
print("Array's memory address:", x_train.data)

print("Shape:", y_train.shape)
print("Data type:", y_train.dtype)
print("Number of dimensions:", y_train.ndim)
print("Size (total number of elements):", y_train.size)
print("Item size (size in bytes of each element):", y_train.itemsize)
print("Array's memory address:", y_train.data)

input_size = len(features)
hidden_Layer_1 = 128
hidden_layer_2 = 64
dropout = 0.45

model = Sequential()
model.add(Dense(hidden_Layer_1, input_dim=input_size, activation = 'relu' ))

model.add(Dropout(dropout))
model.add(Dense(hidden_layer_2, activation = 'relu'))

model.add(Dropout(dropout))
model.add(Dense(num_labels, activation = 'sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['binary_accuracy'])

model.summary()

history = model.fit(x_train, y_train,batch_size=15, epochs=100)

# Perform model evaluation on the test dataset
model.evaluate(x_test, y_test)

