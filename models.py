import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow import keras
from tensorflow.keras import layers, activations
from tensorflow.keras.layers import Dropout


# Implements scoring system for evaluation
def scoringSystem(y_true, y_pred):
    #print(type(y_true))

    weight = tf.ones_like(y_true)
    absolute = tf.math.abs(y_pred - y_true)
    # neuer stand
    #print("absolut: ", absolute)
    counter = tf.math.reduce_sum((weight * 2 * y_true * absolute))
    denominator = tf.math.reduce_sum(weight)
    print(denominator)

    formula = 1e4 - (counter / denominator) * 1e6

    return formula

class simpleRegression(torch.nn.Module):
    def __init__(self, input_size, output_size):
        super(simpleRegression, self).__init__()
        self.l1 = nn.Linear(input_size, output_size)

    def forward(self, x):
        out = self.l1(x)
        return out


class feedForward(torch.nn.Module):
    def __init__(self, input_size, output_size):
        super(feedForward, self).__init__()
        #self.flatten = nn.Flatten()
        self.l1 = nn.Linear(input_size, 150)
        self.relu = nn.ReLU()
        self.l2 = nn.Linear(150, output_size)

    def forward(self, x):
        #out = self.flatten(x)
        out = self.l1(x)
        out = self.relu(out)
        out = self.l2(out)
        return out


def sequentialModel(inputs, lr_rate):
    model = keras.Sequential([
        keras.layers.InputLayer((55, 300)),
        keras.layers.Flatten(),
       # layers.Activation(activations.selu),
       # layers.Dense(2048),
       # layers.Activation(activations.selu),
       # layers.Dense(1024),
        #layers.Activation(activations.selu),
        #layers.Dense(512),
        layers.Activation(activations.selu),
       # layers.Dense(256),
       # layers.Activation(activations.selu),
        layers.Dense(128),
        layers.Activation(activations.selu),
        layers.Dense(64),
        layers.Activation(activations.selu),
        layers.Dense(55)
    ])

    optimizer = keras.optimizers.Adamax()

    model.compile(loss="mse",
                  optimizer=optimizer,
                  metrics=[scoringSystem])

    return model


def sequentialDropout(inputs, lr_rate):
    dropout_rate = 0.3
    model = keras.Sequential([
        layers.Dropout(dropout_rate),
        keras.layers.InputLayer((55, 300)),
        keras.layers.Flatten(),
        #layers.Dropout(dropout_rate),
        layers.Activation(activations.selu),
        layers.Dense(256),
        layers.Activation(activations.selu),
        #layers.Dropout(dropout_rate),
        layers.Dense(128),
        layers.Activation(activations.selu),
        #layers.Dropout(dropout_rate),
        layers.Dense(64),
        layers.Activation(activations.selu),
        #layers.Dropout(dropout_rate),
        #layers.Dense(32),
        #layers.Dropout(dropout_rate),
        layers.Dense(55)
    ])

    optimizer = keras.optimizers.Adam(learning_rate=lr_rate)

    model.compile(loss="mse",
                  optimizer=optimizer,
                  metrics=[scoringSystem])
    return model


def lstmModel(inputs):
    model = keras.Sequential([
        layers.LSTM(inputs),
        layers.LSTM(1)
    ])
    optimizer = keras.optimizers.Adam()

    model.compile(loss="mse",
                  optimizer=optimizer,
                  metrics=[scoringSystem])

    return model


def cnnModel(number_of_measurements):
    model = keras.Sequential([
        keras.layers.InputLayer(input_shape=(55, number_of_measurements)),
        layers.Conv1D(filters=32, kernel_size=5),
        layers.MaxPooling1D(pool_size=2),
        layers.Activation(activations.selu),
        layers.Conv1D(filters=32, kernel_size=5),
        layers.MaxPooling1D(pool_size=3),
        layers.Activation(activations.selu),
        layers.Conv1D(filters=32, kernel_size=5),
        layers.MaxPooling1D(pool_size=3),
        layers.Flatten(),
        layers.Dense(1024),
        layers.Activation(activations.selu),
        layers.Dense(1024),
        layers.Activation(activations.selu),
        layers.Dense(1024),
        layers.Activation(activations.selu),
        layers.Dense(1024),
        layers.Activation(activations.selu),
        layers.Dense(55),
    ])
    optimizer = keras.optimizers.Adam(learning_rate=0.0001)

    model.compile(loss="mse",
                  optimizer=optimizer,
                  metrics=[scoringSystem])

    return model


def cnnModelFilters(number_of_measurements):
    model = keras.Sequential([
        keras.layers.InputLayer(input_shape=(55, number_of_measurements)),
        layers.Conv1D(filters=32, kernel_size=5),
        layers.MaxPooling1D(pool_size=2),
        layers.Activation(activations.selu),
        layers.Conv1D(filters=32, kernel_size=5),
        layers.MaxPooling1D(pool_size=2),
        layers.Activation(activations.selu),
        layers.Conv1D(filters=32, kernel_size=5),
        layers.MaxPooling1D(pool_size=2),
        layers.Flatten(),
        layers.Dense(1024),
        layers.Activation(activations.selu),
        layers.Dense(512),
        layers.Activation(activations.selu),
        layers.Dense(256),
        layers.Activation(activations.selu),
        layers.Dense(128),
        layers.Activation(activations.selu),
        layers.Dense(55),
    ])
    optimizer = keras.optimizers.Adam(learning_rate=0.0001)

    model.compile(loss="mae",
                  optimizer=optimizer,
                  metrics=[scoringSystem])

    return model