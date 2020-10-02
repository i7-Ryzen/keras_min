import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # to ignore warning errors
from load_from_h5.loading_from_h5 import load_model_from_h5
from tensorflow.keras.models import load_model
from pathlib import Path
import pickle
from tests_models import build_models

def pickle_model(dic, new_name):
    with open(new_name, 'wb') as fp:
        pickle.dump(dic, fp, protocol=pickle.HIGHEST_PROTOCOL)



if __name__ == "__main__":
    name_of_model = "resnet50"
    model = getattr(build_models, name_of_model)()
    file_path = Path(__file__).parent.absolute() / "models" / name_of_model / ".h5"
    basic_model = load_model(file_path)
    basic_model.summary()
    # build model
    dic = load_model_from_h5(file_path)
    pickle_model(dic, name_of_model+'_new_storage.p')

    # # print the new model
    # file = open(name_of_model+'_new_storage.p', 'rb')
    # data = pickle.load(file)
    # print(data)