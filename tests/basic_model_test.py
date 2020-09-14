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

def define_model():
    classifier = Sequential()
    # Add a Convolution layer with 32 nodes of type Relu and a MaxPooling2D of size 2x2
    classifier.add(Convolution2D(32, (3, 3), input_shape=(200, 200, 3), activation="relu", kernel_initializer=initializers.RandomNormal(stddev=0.01), bias_initializer=initializers.Zeros()))
    classifier.add(MaxPooling2D(pool_size=(2, 2)))
    # Flatten the first layer
    classifier.add(Flatten())
    # We add layers with Relu as activation type, and units number of nodes.
    classifier.add(Dense(activation='relu', units=128, kernel_initializer=initializers.RandomNormal(stddev=0.01), bias_initializer=initializers.Zeros()))
    classifier.add(Dense(activation='relu', units=64, kernel_initializer=initializers.RandomNormal(stddev=0.01), bias_initializer=initializers.Zeros()))
    classifier.add(Dense(activation='relu', units=32, kernel_initializer=initializers.RandomNormal(stddev=0.01), bias_initializer=initializers.Zeros()))
    classifier.add(Dense(activation='sigmoid', units=1))
    # We compile our model classifier
    classifier = Model(inputs=[classifier.input], outputs=[classifier.output])
    classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return classifier


basic_model = define_model()
basic_model.summary()
basic_model.save("model.h5")


if __name__ == "__main__":
    file_path = Path(__file__).parent.absolute()/"model.h5"
    basic_model = load_model(file_path)
    basic_model.summary()


    # build model
    dic = load_model_from_h5(file_path)
    deploy = Deploy(dic)

    # image data
    dataset_home = Path(__file__).parent.absolute() / 'test_img/'
    image_name = next(iter(listdir(dataset_home)[:3]))
    image = Image.open(dataset_home / image_name)
    new_image = image.resize((200, 200))
    x = np.asarray(new_image).reshape(1,3,200,200)*(1./255)
    x_reshaped = np.moveaxis(x, 1, -1)

    avg_time = [0, 0]
    outs = [[], []]
    n_simulations = 1
    for _ in range(n_simulations):

        # our methods
        t1 = time.time()
        o1 = deploy(x_reshaped)
        avg_time[0] += (time.time() - t1)/n_simulations
        outs[0].append(o1)

        # Keras
        t1 = time.time()
        o2 = basic_model.predict(x_reshaped)
        avg_time[1] += (time.time() - t1)/n_simulations
        outs[1].append(o2)

    print("- predictions of our method", outs[0])
    print("- predictions of keras method", outs[1])
    print("- difference of predictions ", [(o1 - o2).sum() for o1, o2 in zip(*outs)])
    print("- the average run time of keras: ", avg_time[1], "the average run time  of our implementation: ", avg_time[0])
    print('- Ratio speed: (keras/our_implementation)', avg_time[1] / avg_time[0])


# ################################################
# result for model.h5 (from assignment 1)
# ################################################
#
# - predictions of our method [array([[0.56942517]]), array([[0.56942517]]), array([[0.56942517]]), array([[0.56942517]]), array([[0.56942517]])]
# - predictions of keras method [array([[0.5694265]], dtype=float32), array([[0.5694265]], dtype=float32), array([[0.5694265]], dtype=float32), array([[0.5694265]], dtype=float32), array([[0.5694265]], dtype=float32)]
# - difference of predictions  [-1.307866071664776e-06, -1.307866071664776e-06, -1.307866071664776e-06, -1.307866071664776e-06, -1.307866071664776e-06]
# - the average run time of keras:  0.06406269073486329 the average run time  of our implementation:  0.10188374519348145
# - Ratio speed: (our_implementation/keras) 1.590375677711516

