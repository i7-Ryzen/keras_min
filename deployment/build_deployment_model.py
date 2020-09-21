from activations import activations
from custom_layers.convolution import conv_forward_strides
from custom_layers.zeropadding2d import zeropadding2d
from custom_layers.batchnormalization import bachnormalization
from custom_layers.Activation import relu
from load_from_h5.loading_from_h5 import load_model_from_h5
from custom_layers.pooling import pool2d
from custom_layers.dense import dense_layer
from custom_layers.dense import flatten_layer
from custom_layers.add import add
from os import listdir
from PIL import Image
import numpy as np


def layer_activation(x):
    out = relu(x)
    return out


def layer_bn(x, dict_bn):
    # beta: 0
    # gamma: 0
    # moving_mean: 0
    # moving_variance: 0
    bias = dict_bn[0][3]
    gamma = dict_bn[0][2]
    mean = dict_bn[0][1]
    std = dict_bn[0][0]
    epsilon = dict_bn[1]["config"]["epsilon"]
    out = bachnormalization(x, gamma, bias, mean, std, epsilon)
    if "activation" in dict_bn[1]["config"].keys():
        activation = dict_bn[1]["config"]["activation"]
        method_to_call = getattr(activations, activation)
        output = method_to_call(out)
        return output
    return out

def layer_zeropadding2d(x, dict_zeropad):
    p = dict_zeropad[1]["config"]["padding"][0][0]
    out = zeropadding2d(x, p)
    if "activation" in dict_zeropad[1]["config"].keys():
        activation = dict_zeropad[1]["config"]["activation"]
        method_to_call = getattr(activations, activation)
        output = method_to_call(out)
        return output
    return out


def layer_conv(x, dict_conv):

    w = dict_conv[0][0]
    if dict_conv[1]["config"]["use_bias"] == True:
        b = dict_conv[0][1]
    else :
        b = 0

    p = int(dict_conv[1]["config"]["padding"] == "same")
    s = dict_conv[1]["config"]["strides"][0]
    # print(w.shape,b.shape, p, s)
    # exit()
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

def layer_add(x1, x2, dict_add):
    out = add(x1, x2)
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
    elif dic[1]["class_name"] == "Activation":
        return layer_activation(x)
    elif "bn" in dic[1]["name"] and dic[1]["class_name"] == "BatchNormalization":
        return layer_bn(x, dic)
    elif "pad" in dic[1]["name"] and dic[1]["class_name"] == "ZeroPadding2D":
        return layer_zeropadding2d(x, dic)
    elif "pool" in dic[1]["name"] and dic[1]["class_name"] == "MaxPooling2D":
        return layer_pool(x, dic)
    elif "flatten" in dic[1]["name"] and dic[1]["class_name"] == "Flatten":
        return layer_flatten(x, dic)
    elif "dense" in dic[1]["name"] and dic[1]["class_name"] == "Dense":
        return layer_dense(x, dic)
    return x


def compute_layer_add(x1, x2, dic):
    return layer_add(x1, x2, dic)



# def rum_model(x, dic):
#     for layer_name in dic.keys():
#         x = compute_layer(x, dic[layer_name])
#     return x

def rum_model(x, dic):

    # for resnet and vgg and basic model

    dict_layers = {}
    layer_name_init = list(dic.keys())[0]
    x = compute_layer(x, dic[layer_name_init])
    dict_layers[layer_name_init] = x

    for layer_name in list(dic.keys())[1:]:
        connected_to = dic[layer_name][1]["inbound_nodes"]
        # print(layer_name)
        # print(connected_to)
        # print("khra")
        # print(dic[layer_name])

        if len(connected_to[0]) == 1:
            connected_to_name = connected_to[0][0][0]
            # print(dict_layers[connected_to_name], dic[layer_name])
            x = compute_layer(dict_layers[connected_to_name], dic[layer_name])
            dict_layers[layer_name] = x

        else:
            connected_to_name_1 = connected_to[0][0][0]
            connected_to_name_2 = connected_to[0][1][0]
            x = compute_layer_add(dict_layers[connected_to_name_1], dict_layers[connected_to_name_2], dic[layer_name])
            dict_layers[layer_name] = x

    return x


class Deploy:
    def __init__(self, dic):
        self.dic = dic
    def __call__(self, x):
        # return rum_model(x, self.dic)
        return rum_model(x, self.dic)



if __name__ == "__main__":
    import os
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # to ignore warning errors
    import time
    from tensorflow.keras.models import load_model
    from pathlib import Path

    # load model from keras for test
    file_path = Path(__file__).parent.parent.absolute() / "tests" / "resnet50_model.h5"
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
    n_simulations = 1
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



