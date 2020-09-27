import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # to ignore warning errors
from os import listdir
from deployment.build_deployment_model import Deploy
from load_from_h5.loading_from_h5 import load_model_from_h5
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
import time
from pathlib import Path
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Convolution2D
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.initializers import Constant
from tensorflow.keras.models import Model
from tensorflow.keras import initializers
from tests.save_dictionnary import pickle_model
import matplotlib.pyplot as plt
import seaborn as sns

def define_model_conv(m, kernel_size, number_filter):
    classifier = Sequential()
    classifier.add(Input((200, 200, 3)))
    # Flatten the first layer
    # classifier.add(Flatten())
    classifier.add(Convolution2D(number_filter, (kernel_size, kernel_size), input_shape=(200, 200, 3), activation="relu",
                                 kernel_initializer=initializers.RandomNormal(stddev=0.01),
                                 bias_initializer=initializers.Zeros()))
    # We add layers with Relu as activation type, and units number of nodes.
    for _ in range(m):
        classifier.add(Convolution2D(number_filter, (kernel_size, kernel_size), activation="relu",
                                     kernel_initializer=initializers.RandomNormal(stddev=0.01),
                                     bias_initializer=initializers.Zeros()))
    classifier.add(Flatten())
    classifier.add(Dense(activation='sigmoid', units=1))
    # We compile our model classifier
    classifier = Model(inputs=[classifier.input], outputs=[classifier.output])
    classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return classifier


def define_model_dense(m, number_nodes):
    classifier = Sequential()
    # Flatten the first layer
    classifier.add(Input((200, 200, 3)))
    classifier.add(Flatten())
    # We add layers with Relu as activation type, and units number of nodes.
    for _ in range(m):
        classifier.add(Dense(activation='relu', units=number_nodes, kernel_initializer=initializers.RandomNormal(stddev=0.01),
                             bias_initializer=initializers.Zeros()))

    classifier.add(Dense(activation='sigmoid', units=1))
    # We compile our model classifier
    classifier = Model(inputs=[classifier.input], outputs=[classifier.output])
    classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return classifier


def define_model_pool(m):
    classifier = Sequential()
    classifier.add(Input((200, 200, 3)))

    # Flatten the first layer
    # classifier.add(Flatten())
    # We add layers with Relu as activation type, and units number of nodes.
    for _ in range(m):
        classifier.add(MaxPooling2D(pool_size=(2, 2)))

    classifier.add(Dense(activation='sigmoid', units=1))
    # We compile our model classifier
    classifier = Model(inputs=[classifier.input], outputs=[classifier.output])
    classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return classifier