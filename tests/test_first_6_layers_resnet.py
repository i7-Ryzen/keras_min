import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # to ignore warning errors
from os import listdir
from deployment.build_deployment_model import Deploy
from load_from_h5.loading_from_h5 import load_model_from_h5
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Input
from tensorflow.keras.optimizers import SGD
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.models import load_model



if __name__ == "__main__":
    import os
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # to ignore warning errors
    import time
    from tensorflow.keras.models import load_model
    from pathlib import Path

    # load model from keras for test
    file_path = Path(__file__).parent.parent.absolute() / "tests" / "resnet50_model.h5"
    model = load_model(file_path)
    model_6_firsts = Model(inputs=[model.input], outputs=[Sequential(model.layers[:5]).output])
    model_6_firsts.summary()
    model_6_firsts.save("model_6_firsts.h5")
    # print(model_6_firsts.get_layer("conv1_bn").get_config())
    # print(model_6_firsts.get_layer("conv1_bn").get_weights())


    # build model with our method
    file_path = Path(__file__).parent.parent.absolute() / "tests" / "model_6_firsts.h5"
    dic = load_model_from_h5(file_path)
    # print(dic["conv1_bn"])
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

    print("- predictions shape of our method", outs[0][0].shape)
    print("- predictions shape of keras method", outs[1][0].shape)
    print("- difference of predictions ", [(o1 - o2).sum() for o1, o2 in zip(*outs)])
    print("- the average run time of keras: ", avg_time[1], "the average run time  of our implementation: ", avg_time[0])
    print('- Ratio speed: (our_implementation/keras)', avg_time[0] / avg_time[1])
