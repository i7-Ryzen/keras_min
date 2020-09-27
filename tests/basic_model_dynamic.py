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
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Model
from tensorflow.keras import initializers
from tests.save_dictionnary import pickle_model
import matplotlib.pyplot as plt
import seaborn as sns
import psutil
import sys
from tensorflow.keras import backend as K


def define_model():
    classifier = Sequential()
    # Add a Convolution layer with 32 nodes of type Relu and a MaxPooling2D of size 2x2
    classifier.add(Convolution2D(32, (3, 3), input_shape=(200, 200, 3), activation="relu", kernel_initializer=initializers.RandomNormal(stddev=0.01), bias_initializer=initializers.Zeros()))
    classifier.add(MaxPooling2D(pool_size=(2, 2)))
    # Flatten the first layer
    classifier.add(Flatten())
    # We add layers with Relu as activation type, and units number of nodes.

    classifier.add(Dense(activation='relu', units=1024, kernel_initializer=initializers.RandomNormal(stddev=0.01),
                         bias_initializer=initializers.Zeros()))
    # classifier.add(Dense(activation='relu', units=1024, kernel_initializer=initializers.RandomNormal(stddev=0.01),
    #                      bias_initializer=initializers.Zeros()))
    # classifier.add(Dense(activation='relu', units=1024, kernel_initializer=initializers.RandomNormal(stddev=0.01),
    #                      bias_initializer=initializers.Zeros()))
    # classifier.add(Dense(activation='relu', units=1024, kernel_initializer=initializers.RandomNormal(stddev=0.01),
    #                      bias_initializer=initializers.Zeros()))
    # classifier.add(Dense(activation='relu', units=1024, kernel_initializer=initializers.RandomNormal(stddev=0.01),
    #                      bias_initializer=initializers.Zeros()))
    # classifier.add(Dense(activation='relu', units=1024, kernel_initializer=initializers.RandomNormal(stddev=0.01),
    #                      bias_initializer=initializers.Zeros()))
    # classifier.add(Dense(activation='relu', units=1024, kernel_initializer=initializers.RandomNormal(stddev=0.01),
    #                      bias_initializer=initializers.Zeros()))
    # classifier.add(Dense(activation='relu', units=1024, kernel_initializer=initializers.RandomNormal(stddev=0.01),
    #                      bias_initializer=initializers.Zeros()))
    # classifier.add(Dense(activation='relu', units=1024, kernel_initializer=initializers.RandomNormal(stddev=0.01),
    #                      bias_initializer=initializers.Zeros()))
    # classifier.add(Dense(activation='relu', units=1024, kernel_initializer=initializers.RandomNormal(stddev=0.01),
    #                      bias_initializer=initializers.Zeros()))
    # classifier.add(Dense(activation='relu', units=1024, kernel_initializer=initializers.RandomNormal(stddev=0.01),
    #                      bias_initializer=initializers.Zeros()))
    # classifier.add(Dense(activation='relu', units=1024, kernel_initializer=initializers.RandomNormal(stddev=0.01),
    #                      bias_initializer=initializers.Zeros()))
    # classifier.add(Dense(activation='relu', units=1024, kernel_initializer=initializers.RandomNormal(stddev=0.01),
    #                      bias_initializer=initializers.Zeros()))
    # classifier.add(Dense(activation='relu', units=128, kernel_initializer=initializers.RandomNormal(stddev=0.01),
    #                      bias_initializer=initializers.Zeros()))
    # classifier.add(Dense(activation='relu', units=64, kernel_initializer=initializers.RandomNormal(stddev=0.01),
    #                      bias_initializer=initializers.Zeros()))
    # classifier.add(Dense(activation='relu', units=32, kernel_initializer=initializers.RandomNormal(stddev=0.01),
    #                      bias_initializer=initializers.Zeros()))
    classifier.add(Dense(activation='sigmoid', units=1))
    # We compile our model classifier
    classifier = Model(inputs=[classifier.input], outputs=[classifier.output])
    classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return classifier

print(psutil.virtual_memory().percent)
basic_model = define_model()
print(psutil.virtual_memory().percent)
# basic_model.summary()
# basic_model.save("model.h5")

import gc
gc.collect()
K.clear_session()
del basic_model
import psutil
print(psutil.virtual_memory().percent)

print("##"*10)

if __name__ == "__main__":

    print(psutil.virtual_memory().percent)
    file_path = Path(__file__).parent.absolute()/"model.h5"
    basic_model = load_model(file_path)
    # basic_model.summary()
    print(psutil.virtual_memory().percent)



    # build model
    print(psutil.virtual_memory().percent)
    dic = load_model_from_h5(file_path)
    pickle_model(dic, "new_model.p")
    deploy = Deploy(dic)
    print(psutil.virtual_memory().percent)

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
    n_simulations = 3
    print(psutil.virtual_memory().percent)
    print("##" * 10)
    for _ in range(n_simulations):
        # # our methods
        t1_1 = time.time()
        dic = load_model_from_h5(file_path)
        t1_2 = time.time()
        print(psutil.virtual_memory().percent)

        deploy = Deploy(dic)
        o1 = deploy(x_reshaped)
        temp_1 = (time.time() - t1_1)
        temp_2 = (time.time() - t1_2)
        avg_time[0] += temp_1
        time_list_1[0].append(temp_1)
        time_list_2[0].append(temp_2)
        print("our methods runtime")
        # print(temp_1,temp_2)
        outs[0].append(float((o1)))
        print(psutil.virtual_memory().percent)

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
        # print(temp_1,temp_2)
        outs[1].append(float((o2)))


    print("- predictions of our method", outs[0])
    print("- predictions of keras method", outs[1])
    # print("- difference of predictions ", [(o1 - o2).sum() for o1, o2 in zip(*outs)])
    print("- the average run time of keras: ", avg_time[1]/n_simulations, "the average run time  of our implementation: ", avg_time[0]/n_simulations)
    print('- Ratio speed: (keras/our_implementation)', avg_time[1] / avg_time[0])




    # ################################################
    # plot1 : the distrubution of runtime with loading model time
    # ################################################
    dictio_with = {"Numpy":time_list_1[0], "keras":time_list_1[1]}
    for name in dictio_with.keys():
        # Draw the density plot
        sns.distplot(dictio_with[name], hist=True, kde=True,
                     kde_kws={'linewidth': 3},
                     label= name)

    # Plot formatting
    plt.legend(prop={'size': 7})
    plt.title('basic_model : runtime density with loading the model')
    plt.xlabel('runtime (s)')
    plt.ylabel('Density')
    plt.savefig(Path(__file__).parent.absolute() / "plots" /  "basic_model_with_loading")
    plt.show()


    # ################################################
    # plot2 : the distrubution of runtime without loading model time
    # ################################################
    dictio_without = {"Numpy":time_list_2[0], "keras":time_list_2[1]}

    for name in dictio_without.keys():
        # Draw the density plot
        sns.distplot(dictio_without[name], hist=True, kde=True,
                     kde_kws={'linewidth': 3},
                     label= name)

    # Plot formatting
    plt.legend(prop={'size': 7})
    plt.title('basic_model : runtime density without loading the model')
    plt.xlabel('runtime (s)')
    plt.ylabel('Density')
    plt.savefig(Path(__file__).parent.absolute() / "plots" /  "basic_model_without_loading")
    plt.show()

# ################################################
# result for model.h5 (from assignment 1)
# ################################################
#
# our methods runtime
# 0.17354178428649902
# keras runtime
# 0.6074240207672119
# our methods runtime
# 0.15387845039367676
# keras runtime
# 0.5602781772613525
# our methods runtime
# 0.19624710083007812
# keras runtime
# 0.4925360679626465
# our methods runtime
# 0.1520984172821045
# keras runtime
# 0.563908576965332
# our methods runtime
# 0.19113993644714355
# keras runtime
# 0.46692824363708496
# our methods runtime
# 0.2942178249359131
# keras runtime
# 0.47580552101135254
# our methods runtime
# 0.22774314880371094
# keras runtime
# 0.5193436145782471
# our methods runtime
# 0.1737666130065918
# keras runtime
# 0.8267509937286377
# our methods runtime
# 0.23139119148254395
# keras runtime
# 0.4813826084136963
# our methods runtime
# 0.15049433708190918
# keras runtime
# 0.4474666118621826
# - predictions of our method [array([[0.49987653]]), array([[0.49987653]]), array([[0.49987653]]), array([[0.49987653]]), array([[0.49987653]]), array([[0.49987653]]), array([[0.49987653]]), array([[0.49987653]]), array([[0.49987653]]), array([[0.49987653]])]
# - predictions of keras method [array([[0.49987653]], dtype=float32), array([[0.49987653]], dtype=float32), array([[0.49987653]], dtype=float32), array([[0.49987653]], dtype=float32), array([[0.49987653]], dtype=float32), array([[0.49987653]], dtype=float32), array([[0.49987653]], dtype=float32), array([[0.49987653]], dtype=float32), array([[0.49987653]], dtype=float32), array([[0.49987653]], dtype=float32)]
# - difference of predictions  [1.773440228003409e-09, 1.773440228003409e-09, 1.773440228003409e-09, 1.773440228003409e-09, 1.773440228003409e-09, 1.773440228003409e-09, 1.773440228003409e-09, 1.773440228003409e-09, 1.773440228003409e-09, 1.773440228003409e-09]
# - the average run time of keras:  0.5441648006439208 the average run time  of our implementation:  0.19443130493164062
# - Ratio speed: (keras/our_implementation) 2.7987509564637323




