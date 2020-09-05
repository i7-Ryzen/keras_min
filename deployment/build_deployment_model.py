from custom_layers.convolution import conv_forward_strides
from load_from_h5.loading_from_h5 import load_model_from_h5

file_path = r"/home/soufiane/PycharmProjects/keras_min/model.h5"
dict_2 = load_model_from_h5(file_path)

# print(dict_2.keys())
dict_conv = dict_2['conv2d_3']
print(int(dict_conv[1]["padding"] == "same"))
# print(dict_conv[1])

def compute_layer(x,dict_conv):
    w = dict_conv[0][0]
    b = dict_conv[0][1]
    p = int(dict_conv[1]["padding"] == "same")
    s = dict_conv["strides"][0]
    out = conv_forward_strides(x, w, b, s, p)
    return out






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


