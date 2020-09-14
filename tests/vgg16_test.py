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

    # load model with  keras (for test)
    file_path = Path(__file__).parent.absolute() / "model_vgg16.h5"
    vgg16_model = load_model(file_path)

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

    # deployment

    deploy(x_reshaped)


    avg_time = [0, 0]
    outs = [[], []]
    n_simulations = 10
    for _ in range(n_simulations):
        # our methods
        t1 = time.time()
        o1 = deploy(x_reshaped)
        avg_time[0] += (time.time() - t1)/n_simulations
        outs[0].append(o1)

        # keras
        t1 = time.time()
        o2 = vgg16_model.predict(x_reshaped)
        avg_time[1] += (time.time() - t1)/n_simulations
        outs[1].append(o2)

    print("- predictions of our method", outs[0])
    print("- predictions of keras method", outs[1])
    print("- difference of predictions ", [(o1 - o2).sum() for o1, o2 in zip(*outs)])
    print("- the average run time of keras: ", avg_time[1], "the average run time  of our implementation: ", avg_time[0])
    print('- Ratio speed: (keras/our_implementation)', avg_time[1] / avg_time[0])


#################################################
# result for vgg16.h5
#################################################

# Model: "functional_1"
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
# - predictions of our method [array([[0.31569217]]), array([[0.31569217]]), array([[0.31569217]]), array([[0.31569217]]), array([[0.31569217]])]
# - predictions of keras method [array([[0.31569234]], dtype=float32), array([[0.31569234]], dtype=float32), array([[0.31569234]], dtype=float32), array([[0.31569234]], dtype=float32), array([[0.31569234]], dtype=float32)]
# - difference of predictions  [-1.6569631033913623e-07, -1.6569631033913623e-07, -1.6569631033913623e-07, -1.6569631033913623e-07, -1.6569631033913623e-07]
# - the average run time of keras:  0.22639803886413576 the average run time  of our implementation:  0.5668399333953857
# - Ratio speed: (our_implementation/keras) 2.503731641136491
