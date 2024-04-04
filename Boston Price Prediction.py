import numpy as np
import os
import random
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import SGD

def load_data():
    datafile = './work/housing.data'
    data = np.fromfile(datafile, sep=' ', dtype=np.float32)

    feature_names = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE',
                     'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
    feature_num = len(feature_names)

    data = data.reshape([data.shape[0] // feature_num, feature_num])

    ratio = 0.8
    offset = int(data.shape[0] * ratio)
    training_data = data[:offset]

    maximums, minimums = training_data.max(axis=0), training_data.min(axis=0)

    for i in range(feature_num):
        data[:, i] = (data[:, i] - minimums[i]) / (maximums[i] - minimums[i])

    training_data = data[:offset]
    test_data = data[offset:]
    return training_data, test_data

training_data, test_data = load_data()

train_features = training_data[:, :-1]
train_labels = training_data[:, -1:]

test_features = test_data[:, :-1]
test_labels = test_data[:, -1:]

model = Sequential([
    Dense(1, input_dim=13, activation='linear')
])

model.compile(optimizer=SGD(learning_rate=0.01), loss='mean_squared_error')

model.fit(train_features, train_labels, epochs=100, batch_size=32)

model.evaluate(test_features, test_labels, verbose=2)
