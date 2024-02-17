# Muhammad Raisul Islam Evan
# MBSTU CSE- 15th
# Tangail, Dhaka, Bangladesh


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

data = pd.read_csv('/home/evan/Downloads/final.csv')


data  = data.values


X_train  = np.asarray(data[:, 1:16])
Y_train = np.asarray(data[:, 16])


xtrain, xtest, ytrain, ytest = train_test_split(X_train, Y_train, test_size=0.3)

X_train = np.reshape(X_train, (-1, 1, 15))
Y_train = np.reshape(Y_train, (-1, 1, 1))

model=Sequential()

model.add(LSTM(100, input_shape=(1,15), return_sequences=True))
model.add(LSTM(100, return_sequences=True, recurrent_dropout=0.2))
model.add(LSTM(100, return_sequences=True, recurrent_dropout=0.2))
model.add(LSTM(100, return_sequences=False, recurrent_dropout=0.2))  # Change to return_sequences=False
model.add(Dense(1, activation='linear'))

model.compile(optimizer='adam', loss='mean_squared_error')

X_train = X_train.astype(np.int32)
Y_train = Y_train.astype(np.int32)
model.fit(X_train, Y_train, batch_size=32, epochs=100)


xtest = xtest.astype(np.int32)
xtest = np.reshape(xtest, (-1, 1, 15))

y_pred = model.predict(xtest)
print(y_pred.shape)
print(y_pred)


threshold = 0.6
y_pred = (y_pred > threshold).astype(np.int32)
print(y_pred.shape)
print(y_pred)

ytest = (ytest > threshold).astype(np.int32)
accuracy = accuracy_score(ytest, y_pred)
print("Accuracy:", accuracy*100)
