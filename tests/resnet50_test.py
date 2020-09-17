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
from tests.save_dictionnary import pickle_model
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
    t1 = time.time()
    dic = load_model_from_h5(file_path)
    pickle_model(dic, "new_resnet50.p")
    # print(time.time()-t1)
    deploy = Deploy(dic)


    # image data
    dataset_home = Path(__file__).parent.parent.absolute() / "tests" / 'test_img/'
    image_name = next(iter(listdir(dataset_home)[:3]))
    image = Image.open(dataset_home / image_name)
    new_image = image.resize((200, 200))
    x = np.asarray(new_image).reshape(1,3,200,200)*(1./255)
    x_reshaped = np.moveaxis(x, 1, -1)
    o2 = model_6_firsts.predict(x_reshaped)

    avg_time = [0, 0]
    outs = [[], []]
    n_simulations = 10
    for _ in range(n_simulations):
        # our methods
        t1 = time.time()
        dic = load_model_from_h5(file_path)
        deploy = Deploy(dic)
        o1 = deploy(x_reshaped)
        avg_time[0] += (time.time() - t1)/n_simulations
        print("our methods runtime")
        print(time.time() - t1)
        outs[0].append(o1)

        # keras
        t2 = time.time()
        model_6_firsts = load_model(file_path)
        o2 = model_6_firsts.predict(x_reshaped)
        avg_time[1] += (time.time() - t2)/n_simulations
        print("keras runtime")
        print(time.time() - t2)
        outs[1].append(o2)

    print("- predictions of our method", outs[0])
    print("- predictions of keras method", outs[1])
    print("- difference of predictions ", [(o1 - o2).sum() for o1, o2 in zip(*outs)])
    print("- the average run time of keras: ", avg_time[1], "the average run time  of our implementation: ", avg_time[0])
    print('- Ratio speed: (keras/our_implementation)', avg_time[1] / avg_time[0])


# our methods runtime
# 0.6467864513397217
# keras runtime
# 3.841763973236084
# our methods runtime
# 0.5668859481811523
# keras runtime
# 3.2340502738952637
# our methods runtime
# 0.5699708461761475
# keras runtime
# 3.3078646659851074
# our methods runtime
# 0.5267186164855957
# keras runtime
# 3.3806936740875244
# our methods runtime
# 0.572357177734375
# keras runtime
# 3.452599048614502
# our methods runtime
# 0.5805795192718506
# keras runtime
# 3.44966459274292
# our methods runtime
# 0.5792791843414307
# keras runtime
# 3.154789686203003
# our methods runtime
# 0.56538987159729
# keras runtime
# 3.409698963165283
# our methods runtime
# 0.5203807353973389
# keras runtime
# 3.1763288974761963
# our methods runtime
# 0.5225241184234619
# keras runtime
# 3.504319667816162
# - predictions of our method [array([[0.74037338]]), array([[0.74037338]]), array([[0.74037338]]), array([[0.74037338]]), array([[0.74037338]]), array([[0.74037338]]), array([[0.74037338]]), array([[0.74037338]]), array([[0.74037338]]), array([[0.74037338]])]
# - predictions of keras method [array([[0.74037385]], dtype=float32), array([[0.74037385]], dtype=float32), array([[0.74037385]], dtype=float32), array([[0.74037385]], dtype=float32), array([[0.74037385]], dtype=float32), array([[0.74037385]], dtype=float32), array([[0.74037385]], dtype=float32), array([[0.74037385]], dtype=float32), array([[0.74037385]], dtype=float32), array([[0.74037385]], dtype=float32)]
# - difference of predictions  [-4.723545041773747e-07, -4.723545041773747e-07, -4.723545041773747e-07, -4.723545041773747e-07, -4.723545041773747e-07, -4.723545041773747e-07, -4.723545041773747e-07, -4.723545041773747e-07, -4.723545041773747e-07, -4.723545041773747e-07]
# - the average run time of keras:  3.3911612510681155 the average run time  of our implementation:  0.5650691270828248
# - Ratio speed: (keras/our_implementation) 6.001321057080256