import json
import h5py


def load_model_from_h5(file):
    """
    return  the structure configuration and the weights of HDF5 file.

    Args:
      file (str) : Path to the file to analyze
    """

    #  dict_weights
    f = h5py.File(file, mode='r')
    g = f["model_weights"]
    dict_weights = dict()
    for p_name in g.keys():
        param = g[p_name]
        for i, k_name in enumerate(param.keys()):
            T = tuple()
            for key_ in param.get(k_name).keys():
                x = param.get(k_name).get(key_)[:]
                T = (x,) + T
            dict_weights[p_name] = T

    #  dict_config
    str_model_config = f.attrs["model_config"]
    res = json.loads(str_model_config)["config"]['layers']
    dict_configs = dict()
    for j, x in enumerate(res):
        x_temp = x["config"]
        dict_configs[x_temp["name"]] = x_temp
    keys_weights = list(dict_weights.keys())
    keys_configs = list(dict_configs.keys())
    # merge dict_weights & dict_config
    for key2 in keys_configs:
        if key2 in keys_weights:
            dict_configs[key2] = (dict_weights[key2], dict_configs[key2])
        else:
            dict_configs[key2] = (None, dict_configs[key2])
    f.close()
    return dict_configs


file_path = r"/Users/soufiane/PycharmProjects/keras_min/model.h5"
print(load_model_from_h5(file_path)["conv2d_3"])
