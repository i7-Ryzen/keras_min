import numpy as np

def add(x1,x2):
    return x1+x2

if __name__ == "__main__":
    import os
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # to ignore warning errors
    import time
    import tensorflow.keras.backend as keras
    import tensorflow.keras.layers as layers
    from pathlib import Path
    from tensorflow.keras.models import load_model

    file_path = Path(__file__).parent.parent.absolute() / "tests" / "resnet50_model.h5"
    resnet50_model = load_model(file_path)
    add_keras = resnet50_model.get_layer("conv2_block1_add")

    # # data_type = channels_last
    A1 = np.random.normal(loc=0, scale=1, size=(1, 50, 50, 256))
    A2 = np.random.normal(loc=0, scale=1, size=(1, 50, 50, 256))
    avg_time = [0, 0]
    outs = [[], []]
    x1 = keras.constant(A1)
    x2 = keras.constant(A2)




    n_simulations = 10
    for _ in range(n_simulations):
        # Keras
        t1 = time.time()
        o1 = add_keras([A1, A2]).numpy()
        # print(batchnormalization_keras.get_weights())
        avg_time[0] += (time.time() - t1)/n_simulations
        outs[0].append(o1)

        # our methods
        t1 = time.time()
        o2 = add(A1, A2)
        avg_time[1] += (time.time() - t1)/n_simulations
        outs[1].append(o2)

    print("difference of predictions ", [(o1 - o2).sum() for o1, o2 in zip(*outs)])
    print("the average run time of keras: ", avg_time[0], "the average run time  of our implementation: ", avg_time[1])
    print('Ratio speed: (our_implementation/keras)', avg_time[1] / avg_time[0])

    # print(A1.shape)
    # print(A2.shape)
    # print("#"*10)
    # print(o1)
    # print("#"*10)
    # print(o2)