import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # to ignore warning errors
from os import listdir
from deployment.build_deployment_model import Deploy
from load_from_h5.loading_from_h5 import load_model_from_h5
from tensorflow.keras.applications import ResNet50, ResNet152V2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Flatten
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.layers import Dense
from tests.save_dictionnary import pickle_model
import numpy as np
from PIL import Image



# define the model
def define_model():
    # load model
    ResNet152V2_model = ResNet152V2(include_top=False, weights='imagenet', input_shape=(200, 200, 3))
    # mark loaded layers as not trainable
    output = ResNet152V2_model.layers[-1].output
    output = Flatten()(output)
    # define new model
    resnet50_model = Model(inputs=ResNet152V2_model.inputs, outputs=output)

    for layer in resnet50_model.layers:
        layer.trainable = False

    flat1 = resnet50_model.layers[-1].output
    class1 = Dense(128, activation='relu', kernel_initializer='he_uniform')(flat1)
    output = Dense(1, activation='sigmoid')(class1)

    # define new model
    resnet50_model = Model(inputs=resnet50_model.inputs, outputs=output)

    # compile model
    opt = SGD(lr=0.001, momentum=0.9)
    resnet50_model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])
    return resnet50_model

ResNet152V2_model = define_model()
ResNet152V2_model.summary()
ResNet152V2_model.save("ResNet152V2_model.h5")



if __name__ == "__main__":
    import os
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # to ignore warning errors
    import time
    from tensorflow.keras.models import load_model
    from pathlib import Path
    import matplotlib.pyplot as plt
    import seaborn as sns
    # load model from keras for test
    file_path = Path(__file__).parent.parent.absolute() / "tests" / "ResNet152V2_model.h5"
    ResNet152V2_model_keras = load_model(file_path)


    # build model with our method
    file_path = Path(__file__).parent.parent.absolute() / "tests" / "ResNet152V2_model.h5"
    t1 = time.time()
    dic = load_model_from_h5(file_path)
    pickle_model(dic, "new_ResNet152V2_model.p")
    # print(time.time()-t1)
    deploy = Deploy(dic)


    # image data
    dataset_home = Path(__file__).parent.parent.absolute() / "tests" / 'test_img/'
    image_name = next(iter(listdir(dataset_home)[:3]))
    image = Image.open(dataset_home / image_name)
    new_image = image.resize((200, 200))
    x = np.asarray(new_image).reshape(1,3,200,200)*(1./255)
    x_reshaped = np.moveaxis(x, 1, -1)
    o2 = ResNet152V2_model_keras.predict(x_reshaped)

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
        model_resnet152 = load_model(file_path)
        t2_2 = time.time()
        o2 = model_resnet152.predict(x_reshaped)
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
    plt.title('ResNet152V2_model : runtime density with loading the model')
    plt.xlabel('runtime (s)')
    plt.ylabel('Density')
    plt.savefig(Path(__file__).parent.absolute() / "plots" /  "ResNet152V2_model_with_loading")
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
    plt.title('ResNet152V2_model : runtime density without loading the model')
    plt.xlabel('runtime (s)')
    plt.ylabel('Density')
    plt.savefig(Path(__file__).parent.absolute() / "plots" /  "ResNet152V2_model_without_loading")
    plt.show()





