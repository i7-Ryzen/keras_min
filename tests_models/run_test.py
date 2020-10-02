import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # to ignore warning errors

try:
    user_paths = os.environ['PYTHONPATH'].split(os.pathsep)
    print(user_paths)
except KeyError:
    user_paths = []
    print(user_paths)

from deployment.build_deployment_model import Deploy
from load_from_h5.loading_from_h5 import load_model_from_h5
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
import time
from pathlib import Path
from tests_models import build_models
from os import listdir
from tests_models.show_results import show_results


"""
the possible model names are: 
resnet50,
resnet152v2,
vgg16,
model_dense,
model_conv,
model_pool
"""

# choose the name
n_simulations = 10
name_of_model = "model_pool"


model = getattr(build_models, name_of_model)()
model.summary()
model.save(Path(__file__).parent.absolute() / "models" / (name_of_model + ".h5"))


# image data
dataset_home = Path(__file__).parent.absolute() / 'test_img/'
image_name = next(iter(listdir(dataset_home)[:3]))
image = Image.open(dataset_home / image_name)
new_image = image.resize((200, 200))
x = np.asarray(new_image).reshape(1, 3, 200, 200) * (1. / 255)
x_reshaped = np.moveaxis(x, 1, -1)



keras_lists = ([], [], [], [])
kerasmin_lists = ([], [], [], [])


for _ in range(n_simulations):
    file_path = Path(__file__).parent.absolute() / "models" / (name_of_model + ".h5")

    # kerasmin (numpy)
    t1_1 = time.time()
    dic = load_model_from_h5(file_path)
    t1_2 = time.time()
    deploy = Deploy(dic)
    o1 = deploy(x_reshaped)
    t1_3 = time.time()

    kerasmin_lists[0].append((t1_3 - t1_2)) # predictions_list
    kerasmin_lists[1].append((t1_3 - t1_2)) # predictions_list (duplicate to be compatible with plot functions)
    kerasmin_lists[2].append((t1_2- t1_1)) # load_model_keras
    kerasmin_lists[3].append(float((o1))) # output_keras

    # keras
    t2_1 = time.time()
    model = load_model(file_path)
    t2_2 = time.time()
    o2 = model.predict(x_reshaped)
    t2_3 = time.time()
    o2 = model.predict(x_reshaped)
    t2_4 = time.time()

    keras_lists[0].append((t2_3 - t2_2)) # first_predictions_list
    keras_lists[1].append((t2_4 - t2_3)) # second_predictions_list
    keras_lists[2].append((t2_2- t2_1)) # load_model_keras
    keras_lists[3].append(float((o2))) # output_keras



show_results(keras_lists, kerasmin_lists, name_of_model)

