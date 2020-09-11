from activations import activations
from custom_layers.convolution import conv_forward_strides
from load_from_h5.loading_from_h5 import load_model_from_h5
from custom_layers.pooling import pool2d
from custom_layers.dense import dense_layer
from custom_layers.dense import flatten_layer
from os import listdir
from PIL import Image
import numpy as np



def layer_conv(x, dict_conv):
    w = dict_conv[0][0]
    b = dict_conv[0][1]
    p = int(dict_conv[1]["config"]["padding"] == "same")
    s = dict_conv[1]["config"]["strides"][0]
    out = conv_forward_strides(x, w, b, s, p)
    if "activation" in dict_conv[1]["config"].keys():
        activation = dict_conv[1]["config"]["activation"]
        method_to_call = getattr(activations, activation)
        output = method_to_call(out)
        return output
    return out


def layer_pool(x, dict_pool):
    kernel_size = dict_pool[1]["config"]["pool_size"][0]
    stride = dict_pool[1]["config"]["strides"][0]
    padding = int(dict_pool[1]["config"]["padding"] == "same")
    out = pool2d(x, kernel_size, stride, padding, pool_mode='max')

    if "activation" in dict_pool[1]["config"].keys():
        activation = dict_pool[1]["config"]["activation"]
        method_to_call = getattr(activations, activation)
        out = method_to_call(out)
        return out
    return out


def layer_dense(x, dict_dense):
    w = dict_dense[0][0]
    b = dict_dense[0][1]
    out = dense_layer(x, w, b)
    if "activation" in dict_dense[1]["config"].keys():
        activation = dict_dense[1]["config"]["activation"]
        method_to_call = getattr(activations, activation)
        out = method_to_call(out)
        return out
    return out


def layer_flatten(x, dict_flatten):
    out = flatten_layer(x)
    if "activation" in dict_flatten[1]["config"].keys():
        activation = dict_flatten[1]["config"]["activation"]
        method_to_call = getattr(activations, activation)
        out = method_to_call(out)
        return out
    return out


def compute_layer(x, dic):
    """
    :type x: object
    """

    if "conv" in dic[1]["name"] and dic[1]["class_name"] == "Conv2D":
        return layer_conv(x, dic)
    elif "pool" in dic[1]["name"] and dic[1]["class_name"] == "MaxPooling2D":
        return layer_pool(x, dic)
    elif "flatten" in dic[1]["name"] and dic[1]["class_name"] == "Flatten":
        return layer_flatten(x, dic)
    elif "dense" in dic[1]["name"] and dic[1]["class_name"] == "Dense":
        return layer_dense(x, dic)
    return x


def rum_model(x, dic):
    for layer_name in dic.keys():
        x = compute_layer(x, dic[layer_name])
    return x


class Deploy:
    def __init__(self, dic):
        self.dic = dic
    def __call__(self, x):
        return rum_model(x, self.dic)



if __name__ == "__main__":
    import os
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # to ignore warning errors
    import time
    from tensorflow.keras.models import load_model
    from pathlib import Path

    # load model from keras for test
    file_path = Path(__file__).parent.parent.absolute() / "tests" / "model.h5"
    model = load_model(file_path)
    #model.summary()

    # build model with our method
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
    n_simulations = 5
    for _ in range(n_simulations):
        # Keras
        t1 = time.time()
        o1 = deploy(x_reshaped)
        avg_time[0] += (time.time() - t1)/n_simulations
        outs[0].append(o1)

        # our methods
        t1 = time.time()
        o2 = model.predict(x_reshaped)
        avg_time[1] += (time.time() - t1)/n_simulations
        outs[1].append(o2)

    print("- predictions of our method", outs[0])
    print("- predictions of keras method", outs[1])
    print("- difference of predictions ", [(o1 - o2).sum() for o1, o2 in zip(*outs)])
    print("- the average run time of keras: ", avg_time[1], "the average run time  of our implementation: ", avg_time[0])
    print('- Ratio speed: (our_implementation/keras)', avg_time[0] / avg_time[1])



#################################################
# result for model.h5 (model from assignment1)
#################################################

# - predictions of our method [array([[0.56942517]]), array([[0.56942517]]), array([[0.56942517]]), array([[0.56942517]]), array([[0.56942517]])]
# - predictions of keras method [array([[0.5694265]], dtype=float32), array([[0.5694265]], dtype=float32), array([[0.5694265]], dtype=float32), array([[0.5694265]], dtype=float32), array([[0.5694265]], dtype=float32)]
# - difference of predictions  [-1.307866071664776e-06, -1.307866071664776e-06, -1.307866071664776e-06, -1.307866071664776e-06, -1.307866071664776e-06]
# - the average run time of keras:  0.07912755012512207 the average run time  of our implementation:  0.14931397438049315
# - Ratio speed: (our_implementation/keras) 1.8870036307757203



