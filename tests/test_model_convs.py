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
from utils.utils_tests import  define_model_conv



m = 10
kernel_size=3
number_filter = 8
basic_model = define_model_conv(m=m,kernel_size=kernel_size, number_filter=number_filter)
basic_model.summary()
basic_model.save("model_conv.h5")


if __name__ == "__main__":
    file_path = Path(__file__).parent.absolute()/"model_conv.h5"
    basic_model = load_model(file_path)
    basic_model.summary()


    # build model
    dic = load_model_from_h5(file_path)
    pickle_model(dic, "new_model_conv.p")
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
    n_simulations = 20
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


    print("- predictions of our method", outs[0])
    print("- predictions of keras method", outs[1])
    # print("- difference of predictions ", [(o1 - o2).sum() for o1, o2 in zip(*outs)])
    print("- the average run time of keras: ", avg_time[1]/n_simulations, "the average run time  of our implementation: ", avg_time[0]/n_simulations)
    print('- Ratio speed: (keras/our_implementation)', avg_time[1] / avg_time[0])

    # ################################################
    # plot1_1 : the distrubution of runtime with loading model time
    # ################################################
    dictio_with = {"Numpy":time_list_1[0], "keras":time_list_1[1]}
    for name in dictio_with.keys():
        # Draw the density plot
        sns.distplot(dictio_with[name], hist=True, kde=True,
                     kde_kws={'linewidth': 3},
                     label= name)

    # Plot formatting
    plt.legend(prop={'size': 7})
    plt.title('basic_model_conv ' + str(m) + ' , ' + str(kernel_size) + ' : runtime density with loading the model',loc='center', wrap=True)
    plt.xlabel('runtime (s)')
    plt.ylabel('Density')
    plt.savefig(Path(__file__).parent.absolute() / "plots" /  "basic_model_conv_with_loading")
    plt.show()

    # ################################################
    # plot1_2 : the time serie of runtime with loading model time
    # ################################################
    for name in dictio_with.keys():
        # Draw the density plot
        sns.lineplot(x = np.arange(len(dictio_with[name])) ,y = dictio_with[name], label=name)

    # Plot formatting
    plt.legend(prop={'size': 7})
    plt.title('basic_model_conv ' + str(m) + ' , ' + str(kernel_size) + ' : runtime  with loading the model',
              loc='center', wrap=True)
    plt.xlabel('runtime (s)')
    plt.ylabel('Density')
    plt.savefig(Path(__file__).parent.absolute() / "plots" / "basic_model_conv_with_loading")
    plt.show()


    # ##############################################################
    # plot2_2 : the distrubution of runtime without loading model time
    # ##############################################################
    dictio_without = {"Numpy":time_list_2[0], "keras":time_list_2[1]}

    for name in dictio_without.keys():
        # Draw the density plot
        sns.distplot(dictio_without[name], hist=True, kde=True,
                     kde_kws={'linewidth': 3},
                     label= name)
    # Plot formatting
    plt.legend(prop={'size': 7})
    plt.title('basic_model_conv ' + str(m) + ' , ' + str(kernel_size) +': runtime time serie without loading the model', loc='center', wrap=True)
    plt.xlabel('runtime (s)')
    plt.ylabel('Density')
    plt.savefig(Path(__file__).parent.absolute() / "plots" /  "basic_model_conv_without_loading")
    plt.show()

    # ################################################
    # plot2_2 : the time serie of runtime with loading model time
    # ################################################
    for name in dictio_without.keys():
        # Draw the density plot
        sns.lineplot(x = np.arange(len(dictio_without[name])) ,y = dictio_without[name], label=name)
    # Plot formatting
    plt.legend(prop={'size': 7})
    plt.title('basic_model_conv ' + str(m) + ' , ' + str(kernel_size) +': runtime time serie without loading the model', loc='center', wrap=True)
    plt.xlabel('runtime (s)')
    plt.ylabel('Density')
    plt.savefig(Path(__file__).parent.absolute() / "plots" /  "basic_model_conv_without_loading")
    plt.show()
