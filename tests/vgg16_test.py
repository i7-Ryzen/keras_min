import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # to ignore warning errors
from os import listdir
from deployment.build_deployment_model import Deploy
from load_from_h5.loading_from_h5 import load_model_from_h5
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.optimizers import SGD
import numpy as np
from PIL import Image
from tests.save_dictionnary import pickle_model

from tensorflow.keras.models import load_model

# define the model

def define_model():
    # load model
    model = VGG16(include_top=False, input_shape=(200, 200, 3))
    # mark loaded layers as not trainable
    for layer in model.layers:
        layer.trainable = False
    # add new classifier layers
    flat1 = Flatten()(model.layers[-1].output)
    class1 = Dense(128, activation='relu', kernel_initializer='he_uniform')(flat1)
    output = Dense(1, activation='sigmoid')(class1)
    # define new model
    model = Model(inputs=model.inputs, outputs=output)
    # compile model
    opt = SGD(lr=0.001, momentum=0.9)
    model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])
    return model

model_vgg16 = define_model()
model_vgg16.summary()
model_vgg16.save("model_vgg16.h5")



if __name__ == "__main__":
    import os
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # to ignore warning errors
    import time
    from tensorflow.keras.models import load_model
    from pathlib import Path
    import matplotlib.pyplot as plt
    import seaborn as sns

    # load model with  keras (for test)
    file_path = Path(__file__).parent.absolute() / "model_vgg16.h5"
    vgg16_model = load_model(file_path)

    # build model
    dic = load_model_from_h5(file_path)
    pickle_model(dic, "new_vgg16.p")
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
        model_vgg16 = load_model(file_path)
        t2_2 = time.time()
        o2 = model_vgg16.predict(x_reshaped)
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
    plt.title('vgg16_model : runtime density with loading the model')
    plt.xlabel('runtime (s)')
    plt.ylabel('Density')
    plt.savefig(Path(__file__).parent.absolute() / "plots" /  "vgg16_model_with_loading")
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
    plt.title('vgg16 : runtime density without loading the model')
    plt.xlabel('runtime (s)')
    plt.ylabel('Density')
    plt.savefig(Path(__file__).parent.absolute() / "plots" /  "vgg16_model_without_loading")
    plt.show()

#################################################
# result for vgg16.h5
#################################################

# _________________________________________________________________
# Layer (type)                 Output Shape              Param #
# =================================================================
# input_1 (InputLayer)         [(None, 200, 200, 3)]     0
# _________________________________________________________________
# block1_conv1 (Conv2D)        (None, 200, 200, 64)      1792
# _________________________________________________________________
# block1_conv2 (Conv2D)        (None, 200, 200, 64)      36928
# _________________________________________________________________
# block1_pool (MaxPooling2D)   (None, 100, 100, 64)      0
# _________________________________________________________________
# block2_conv1 (Conv2D)        (None, 100, 100, 128)     73856
# _________________________________________________________________
# block2_conv2 (Conv2D)        (None, 100, 100, 128)     147584
# _________________________________________________________________
# block2_pool (MaxPooling2D)   (None, 50, 50, 128)       0
# _________________________________________________________________
# block3_conv1 (Conv2D)        (None, 50, 50, 256)       295168
# _________________________________________________________________
# block3_conv2 (Conv2D)        (None, 50, 50, 256)       590080
# _________________________________________________________________
# block3_conv3 (Conv2D)        (None, 50, 50, 256)       590080
# _________________________________________________________________
# block3_pool (MaxPooling2D)   (None, 25, 25, 256)       0
# _________________________________________________________________
# block4_conv1 (Conv2D)        (None, 25, 25, 512)       1180160
# _________________________________________________________________
# block4_conv2 (Conv2D)        (None, 25, 25, 512)       2359808
# _________________________________________________________________
# block4_conv3 (Conv2D)        (None, 25, 25, 512)       2359808
# _________________________________________________________________
# block4_pool (MaxPooling2D)   (None, 12, 12, 512)       0
# _________________________________________________________________
# block5_conv1 (Conv2D)        (None, 12, 12, 512)       2359808
# _________________________________________________________________
# block5_conv2 (Conv2D)        (None, 12, 12, 512)       2359808
# _________________________________________________________________
# block5_conv3 (Conv2D)        (None, 12, 12, 512)       2359808
# _________________________________________________________________
# block5_pool (MaxPooling2D)   (None, 6, 6, 512)         0
# _________________________________________________________________
# flatten (Flatten)            (None, 18432)             0
# _________________________________________________________________
# dense (Dense)                (None, 128)               2359424
# _________________________________________________________________
# dense_1 (Dense)              (None, 1)                 129
# =================================================================
# Total params: 17,074,241
# Trainable params: 2,359,553
# Non-trainable params: 14,714,688
# _________________________________________________________________
# our methods runtime
# 0.6663401126861572
# keras runtime
# 0.7601346969604492
# our methods runtime
# 0.7366087436676025
# keras runtime
# 0.46868395805358887
# our methods runtime
# 0.6871731281280518
# keras runtime
# 0.45275235176086426
# our methods runtime
# 0.700108528137207
# keras runtime
# 0.44780778884887695
# our methods runtime
# 0.6964170932769775
# keras runtime
# 0.44745683670043945
# our methods runtime
# 0.6936662197113037
# keras runtime
# 0.4454169273376465
# our methods runtime
# 0.7437295913696289
# keras runtime
# 0.4490516185760498
# our methods runtime
# 0.7004189491271973
# keras runtime
# 0.46564269065856934
# our methods runtime
# 0.6940574645996094
# keras runtime
# 0.44858574867248535
# our methods runtime
# 0.7003269195556641
# keras runtime
# 0.4474794864654541
# - predictions of our method [array([[0.77910069]]), array([[0.77910069]]), array([[0.77910069]]), array([[0.77910069]]), array([[0.77910069]]), array([[0.77910069]]), array([[0.77910069]]), array([[0.77910069]]), array([[0.77910069]]), array([[0.77910069]])]
# - predictions of keras method [array([[0.77910066]], dtype=float32), array([[0.77910066]], dtype=float32), array([[0.77910066]], dtype=float32), array([[0.77910066]], dtype=float32), array([[0.77910066]], dtype=float32), array([[0.77910066]], dtype=float32), array([[0.77910066]], dtype=float32), array([[0.77910066]], dtype=float32), array([[0.77910066]], dtype=float32), array([[0.77910066]], dtype=float32)]
# - difference of predictions  [3.190474306968838e-08, 3.190474306968838e-08, 3.190474306968838e-08, 3.190474306968838e-08, 3.190474306968838e-08, 3.190474306968838e-08, 3.190474306968838e-08, 3.190474306968838e-08, 3.190474306968838e-08, 3.190474306968838e-08]
# - the average run time of keras:  0.4832854032516479 the average run time  of our implementation:  0.7018662929534912
# - Ratio speed: (keras/our_implementation) 0.6885718948233813
