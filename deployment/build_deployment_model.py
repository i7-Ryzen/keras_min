from activations import activations
from custom_layers.convolution import conv_forward_strides
from load_from_h5.loading_from_h5 import load_model_from_h5
from custom_layers.pooling import pool2d
from custom_layers.dense import dense_layer
from custom_layers.dense import flatten_layer




def layer_conv(x, dict_conv):
    w = dict_conv[0][0]
    b = dict_conv[0][1]
    p = int(dict_conv[1]["padding"] == "same")
    s = dict_conv["strides"][0]
    out = conv_forward_strides(x, w, b, s, p)
    if dict_conv[1]["activation"] is not None:
        activation = dict_conv[1]["activation"]
        method_to_call = getattr(activations, activation)
        output = method_to_call(out)
        return output
    return out

def layer_pool(x, dict_pool):
    kernel_size = dict_pool["pool_size"][0]
    stride = dict_pool["strides"][0]
    padding = int(dict_conv[1]["padding"] == "same")
    out = pool2d(x, kernel_size, stride, padding, pool_mode='max')
    if dict_conv[1]["activation"] is not None:
        activation = dict_conv[1]["activation"]
        method_to_call = getattr(activations, activation)
        out = method_to_call(out)
        return out
    return out

def layer_dense(x,dict_dense):
    w = dict_dense[0]
    b = dict_dense[1]
    out = dense_layer(x, w, b)
    if dict_conv[1]["activation"] is not None:
        activation = dict_conv[1]["activation"]
        method_to_call = getattr(activations, activation)
        out = method_to_call(out)
        return out
    return out

def layer_flatten(x,dict_flatten):
    out = flatten_layer(x)
    if dict_conv[1]["activation"] is not None:
        activation = dict_conv[1]["activation"]
        method_to_call = getattr(activations, activation)
        out = method_to_call(out)
        return out
    return out


file_path = r"/Users/soufiane/PycharmProjects/keras_min/model.h5"
dict_2 = load_model_from_h5(file_path)



# def loading():
#     pass
#
#
# class deploy_model(x):
#
#     __init__:
#     pass
#
#     def __call__:
#         for layer in Dict.keys():
#             x = compute_layer(layer_name)
#
#
