import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # to ignore warning errors
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

def define_model(m):
    classifier = Sequential()
    # Add a Convolution layer with 32 nodes of type Relu and a MaxPooling2D of size 2x2
    classifier.add(Convolution2D(32, (3, 3), input_shape=(200, 200, 3), activation="relu", kernel_initializer=initializers.RandomNormal(stddev=0.01), bias_initializer=initializers.Zeros()))
    classifier.add(MaxPooling2D(pool_size=(2, 2)))
    # Flatten the first layer
    classifier.add(Flatten())
    # We add layers with Relu as activation type, and units number of nodes.
    for _ in range(m):
        classifier.add(Dense(activation='relu', units=1024, kernel_initializer=initializers.RandomNormal(stddev=0.01),bias_initializer=initializers.Zeros()))

    classifier.add(Dense(activation='relu', units=128, kernel_initializer=initializers.RandomNormal(stddev=0.01), bias_initializer=initializers.Zeros()))
    classifier.add(Dense(activation='relu', units=64, kernel_initializer=initializers.RandomNormal(stddev=0.01), bias_initializer=initializers.Zeros()))
    classifier.add(Dense(activation='relu', units=32, kernel_initializer=initializers.RandomNormal(stddev=0.01), bias_initializer=initializers.Zeros()))
    classifier.add(Dense(activation='sigmoid', units=1))
    # We compile our model classifier
    classifier = Model(inputs=[classifier.input], outputs=[classifier.output])
    classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return classifier





if __name__ == "__main__":

    def help_f(m):
        name_model = "model" + str(m) + "dense"
        basic_model = define_model(m)
        basic_model.save(name_model + ".h5")


        file_path = Path(__file__).parent.absolute()/"model.h5"
        basic_model = load_model(file_path)
        # basic_model.summary()


        # build model
        dic = load_model_from_h5(file_path)
        pickle_model(dic, "new_model.p")
        deploy = Deploy(dic)

        # image data
        dataset_home = Path(__file__).parent.absolute() / 'test_img/'
        image_name = next(iter(listdir(dataset_home)[:3]))
        image = Image.open(dataset_home / image_name)
        new_image = image.resize((200, 200))
        x = np.asarray(new_image).reshape(1,3,200,200)*(1./255)
        x_reshaped = np.moveaxis(x, 1, -1)

        avg_time = [0, 0]
        time_list_1 = [[], []]
        time_list_2 = [[], []]
        outs = [[], []]
        n_simulations = 5
        for _ in range(n_simulations):
            # our methods
            t1_1 = time.time()
            dic = load_model_from_h5(file_path)
            t1_2 = time.time()
            deploy = Deploy(dic)
            o1 = deploy(x_reshaped)
            temp_1 = (time.time() - t1_1)
            temp_2 = (time.time() - t1_2)
            avg_time[0] += temp_1
            time_list_1[0].append(temp_1)
            time_list_2[0].append(temp_2)
            print("our methods runtime")
            print(temp_1,temp_2)
            outs[0].append(float((o1)))

            # keras
            t2_1 = time.time()
            model_basic = load_model(file_path)
            t2_2 = time.time()
            o2 = model_basic.predict(x_reshaped)
            temp_1 = (time.time() - t2_1)
            temp_2 = (time.time() - t2_2)
            avg_time[1] += temp_1
            time_list_1[1].append(temp_1)
            time_list_2[1].append(temp_2)
            print("keras")
            print(temp_1,temp_2)
            outs[1].append(float((o2)))

        # with_avr_keras =
        # with_avr_ours =
        # return