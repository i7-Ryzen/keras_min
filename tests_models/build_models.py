import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # to ignore warning errors
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Convolution2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Input
from tensorflow.keras import initializers
from tensorflow.keras.applications import ResNet50, ResNet152V2
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.optimizers import SGD


#####################################
# model with convolutions
#####################################
def model_conv(m=50, kernel_size=3, number_filter=32):
    classifier = Sequential()
    classifier.add(Input((200, 200, 3)))
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

#####################################
# model with dense layers
#####################################
def model_dense(m=100, number_nodes=512):
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

#####################################
# model with maxpooling layers
#####################################
def model_pool(m=6):
    classifier = Sequential()
    classifier.add(Input((200, 200, 3)))
    # We add layers with Relu as activation type, and units number of nodes.
    for _ in range(m):
        classifier.add(MaxPooling2D(pool_size=(2, 2)))

    classifier.add(Flatten())
    classifier.add(Dense(activation='sigmoid', units=1))
    # We compile our model classifier
    classifier = Model(inputs=[classifier.input], outputs=[classifier.output])
    classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return classifier

#####################################
# Resnet50
#####################################
def resnet50():
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

#####################################
#  Resnet152v2
#####################################
def resnet152v2():
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

#####################################
#  VGG16
#####################################
def vgg16():
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

