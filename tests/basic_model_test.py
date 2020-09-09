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
    n_simulations = 5
    for _ in range(n_simulations):
        # Keras
        t1 = time.time()
        o1 = deploy(x_reshaped)
        avg_time[0] += (time.time() - t1)/n_simulations
        outs[0].append(o1)

        # our methods
        t1 = time.time()
        o2 = basic_model.predict(x_reshaped)
        avg_time[1] += (time.time() - t1)/n_simulations
        outs[1].append(o2)

    print("- predictions of our method", outs[0])
    print("- predictions of keras method", outs[1])
    print("- difference of predictions ", [(o1 - o2).sum() for o1, o2 in zip(*outs)])
    print("- the average run time of keras: ", avg_time[1], "the average run time  of our implementation: ", avg_time[0])
    print('- Ratio speed: (our_implementation/keras)', avg_time[0] / avg_time[1])


#################################################
# result for model.h5 (from assignment 1)
#################################################

# - predictions of our method [array([[0.56942517]]), array([[0.56942517]]), array([[0.56942517]]), array([[0.56942517]]), array([[0.56942517]])]
# - predictions of keras method [array([[0.5694265]], dtype=float32), array([[0.5694265]], dtype=float32), array([[0.5694265]], dtype=float32), array([[0.5694265]], dtype=float32), array([[0.5694265]], dtype=float32)]
# - difference of predictions  [-1.307866071664776e-06, -1.307866071664776e-06, -1.307866071664776e-06, -1.307866071664776e-06, -1.307866071664776e-06]
# - the average run time of keras:  0.06406269073486329 the average run time  of our implementation:  0.10188374519348145
# - Ratio speed: (our_implementation/keras) 1.590375677711516

