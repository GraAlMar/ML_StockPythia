#
# Forecast of stock movement with Keras and LSTMs (LSTM = Long-Short-Term-Memory) in TensorFlow
#

import tensorflow as tf
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import os
import math

from tensorflow.keras.models import load_model, Sequential
from tensorflow.keras.layers import Conv2D, MaxPool2D, Flatten, Dense, LSTM
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import SGD
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

matplotlib.use('TkAgg')

# source: https://www.macrotrends.net/stocks/charts/TSLA/tesla/stock-price-history
# (14 years stock price history, downloadable as csv file)
# columns:
# 'date' 'open' 'high' 'low' 'close' 'volume'

CSV_FILE = "../data/TSL.csv"
DAYS_BEFORE = 20  # amount of days from the past

initial_stock_data = np.loadtxt(CSV_FILE, delimiter=",", skiprows=12, usecols=(4), comments="#", dtype=float)
# only column no. 4 ("close") will be used
initial_stock_data = np.array(initial_stock_data, dtype="float").reshape(-1, 1)

# normalize
min_max_scaler = MinMaxScaler(feature_range=(0, 1))
stock_data = min_max_scaler.fit_transform(initial_stock_data)


# reorganize
def arrange_data(data, days):
    days_before_values = []  # T- days
    days_values = []  # T
    for i in range(len(data) - days - 1):
        days_before_values.append(data[i:(i + days)])
        days_values.append(data[i + days])
    return np.array(days_before_values), np.array(days_values)


def split_to_percentage(data, percentage):
    return data[0: int(len(data) * percentage)], data[int(len(data) * percentage):]


days_before_values, days_values = arrange_data(stock_data, DAYS_BEFORE)
days_before_values = days_before_values.reshape((days_before_values.shape[0], DAYS_BEFORE, 1))

# splitting of data into 80% for training and the rest as test data
X_train, X_test = split_to_percentage(days_before_values, 0.8)
Y_train, Y_test = split_to_percentage(days_values, 0.8)

# define the keras model
stock_model = Sequential()
stock_model.add(LSTM(10, input_shape=(DAYS_BEFORE, 1), return_sequences=True))
stock_model.add(LSTM(5, activation="relu"))
return_sequences = True

stock_model.add(Dense(1))

sgd = SGD(lr=0.01)

stock_model.summary()
stock_model.compile(loss="mean_squared_error", optimizer=sgd, metrics=[tf.keras.metrics.mse])
stock_model.fit(X_train, Y_train, epochs=100, verbose=1)

# saving the model
stock_model.save("keras_stock.keras")

# evaluation of test data
score, _ = stock_model.evaluate(X_test, Y_test)
rmse = math.sqrt(score)
print("RMSE {}".format(rmse))

# forecast with the test-dataset
predictions_on_test = stock_model.predict(X_test)
predictions_on_test = min_max_scaler.inverse_transform(predictions_on_test)

# forecast with the training-dataset
predictions_on_training = stock_model.predict(X_train)
predictions_on_training = min_max_scaler.inverse_transform(predictions_on_training)

# shift to the right, so that the test curve continues after the training curve
shift = range(len(predictions_on_training) - 1, len(stock_data) - 1 - DAYS_BEFORE - 1)

# Anzeige der Kurven mit matplotlib
plt.plot(initial_stock_data, color="#CFCEC4", label="real movement")
plt.plot(predictions_on_training, label="training", color="green")
plt.plot(shift, predictions_on_test, label="test", color="red", dashes=[6, 2])
plt.legend(loc='upper left')
plt.show()
