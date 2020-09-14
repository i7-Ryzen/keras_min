import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # to ignore warning errors
from os import listdir
from deployment.build_deployment_model import Deploy
from load_from_h5.loading_from_h5 import load_model_from_h5
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Flatten
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.layers import Dense
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model

# define the model

def define_model():
    # load model
    resnet50_model = ResNet50(include_top=False, weights='imagenet', input_shape=(200, 200, 3))
    # mark loaded layers as not trainable
    output = resnet50_model.layers[-1].output
    output = Flatten()(output)
    # define new model
    resnet50_model = Model(inputs=resnet50_model.inputs, outputs=output)

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

resnet50_model = define_model()
resnet50_model.summary()
resnet50_model.save("resnet50_model.h5")

# layer_names=[layer.name for layer in resnet50_model.layers]
# layer_output = resnet50_model.get_layer("conv1_bn")
# weights_bn = layer_output.get_weights()
#
# gamma, beta, mean, std = weights_bn[0], weights_bn[1], weights_bn[2], weights_bn[3]
