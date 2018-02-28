import numpy as np
import tensorflow as tf
import keras as kr
import pandas as pd
from sklearn.model_selection import train_test_split

# load keras packages
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers import Embedding
from keras.layers import Conv1D, GlobalMaxPooling1D
from keras.layers import Conv2D
from keras.layers import MaxPooling1D, LSTM, BatchNormalization, GlobalAveragePooling1D
from keras.datasets import imdb
from keras import optimizers
from keras.callbacks import TensorBoard

from keras import backend as K


def r2_keras(y_true, y_pred):
    SS_res = K.sum(K.square(y_true - y_pred))
    SS_tot = K.sum(K.square(y_true - K.mean(y_true)))
    return (1 - SS_res / (SS_tot + K.epsilon()))



# unzip all the data into futuredata folder
futuredata = pd.read_csv('futuredata/beanpulp_day.csv')
window_width = 30
len_horizon = 3
stride = 3
batch_size = 300
epochs = 1000

filters = 20
kernel_size = 4

x_all = []
y_all = []

for i in range(0, futuredata.shape[0] - window_width - len_horizon, stride):
    # for i in range(0, 100, stride):
    tmp_x = futuredata.loc[i:(i + window_width - 1),
                           ['open', 'close', 'high', 'low', 'volume']]
    x_all.append(np.array(tmp_x))
    tmp_y = futuredata.loc[(i + window_width):(i + window_width +
                                               len_horizon - 1), ['open', 'close', 'high', 'low']]
    y_all.append(np.mean(np.mean(tmp_y)))
    print(i)

x_all_np = np.array(x_all)
y_all_np = np.array(y_all)

x_train, x_test, y_train, y_test = train_test_split(
    x_all_np, y_all_np, test_size=0.1, random_state=100)


model = Sequential()
# LSTM
model.add(LSTM(8, return_sequences=False, input_shape=(window_width, 5)))
# model.add(GlobalAveragePooling1D())
model.add(Dense(1))
model.summary()

#model.compile(loss='mean_squared_error', optimizer='rmsprop', metrics=['mae'])
model.compile(loss='mean_squared_error',
              optimizer='rmsprop', metrics=[r2_keras])

model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          validation_data=(x_test, y_test))
