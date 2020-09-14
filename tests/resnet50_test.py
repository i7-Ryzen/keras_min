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



if __name__ == "__main__":
    import os
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # to ignore warning errors
    import time
    from tensorflow.keras.models import load_model
    from pathlib import Path

    # load model from keras for test
    file_path = Path(__file__).parent.parent.absolute() / "tests" / "resnet50_model.h5"
    model_6_firsts = load_model(file_path)


    # build model with our method
    file_path = Path(__file__).parent.parent.absolute() / "tests" / "resnet50_model.h5"
    dic = load_model_from_h5(file_path)
    deploy = Deploy(dic)


    # image data
    dataset_home = Path(__file__).parent.parent.absolute() / "tests" / 'test_img/'
    image_name = next(iter(listdir(dataset_home)[:3]))
    image = Image.open(dataset_home / image_name)
    new_image = image.resize((200, 200))
    x = np.asarray(new_image).reshape(1,3,200,200)*(1./255)
    x_reshaped = np.moveaxis(x, 1, -1)



    avg_time = [0, 0]
    outs = [[], []]
    n_simulations = 1
    for _ in range(n_simulations):
        # Keras
        t1 = time.time()
        o1 = deploy(x_reshaped)
        avg_time[0] += (time.time() - t1)/n_simulations
        outs[0].append(o1)

        # our methods
        t1 = time.time()
        o2 = model_6_firsts.predict(x_reshaped)
        avg_time[1] += (time.time() - t1)/n_simulations
        outs[1].append(o2)

    print("- predictions shape of our method", outs[0][0])
    print("- predictions shape of keras method", outs[1][0])
    print("- difference of predictions ", [(o1 - o2).sum() for o1, o2 in zip(*outs)])
    print("- the average run time of keras: ", avg_time[1], "the average run time  of our implementation: ", avg_time[0])
    print('- Ratio speed: (our_implementation/keras)', avg_time[0] / avg_time[1])
