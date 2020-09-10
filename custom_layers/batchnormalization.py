import numpy as np


def bachnormalization(x, gamma, bias, mean, std, epsilon):

    normalized = (x - mean.reshape(1,1,1,-1)) / np.sqrt(std.reshape(1,1,1,-1) + epsilon)
    output = gamma.reshape(1,1,1,-1) * normalized + bias.reshape(1,1,1,-1)
    return output



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
    batchnormalization_keras = resnet50_model.get_layer("conv1_bn")
    weights_bn = batchnormalization_keras.get_weights()
    gamma, beta, mean, std = weights_bn[0], weights_bn[1], weights_bn[2], weights_bn[3]
    epsilon = 1.001e-05


    # # data_type = channels_last
    A = np.random.normal(loc=10, scale=1, size=(1, 100, 100, 64))
    avg_time = [0, 0]
    outs = [[], []]
    x = keras.constant(A)



    n_simulations = 10
    for _ in range(n_simulations):
        # Keras
        t1 = time.time()
        o1 = batchnormalization_keras(x).numpy()
        # print(batchnormalization_keras.get_weights())
        avg_time[0] += (time.time() - t1)/n_simulations
        outs[0].append(o1)

        # our methods
        t1 = time.time()
        o2 = bachnormalization(A, gamma, beta, mean, std, epsilon)
        avg_time[1] += (time.time() - t1)/n_simulations
        outs[1].append(o2)

    print("difference of predictions ", [(o1 - o2).sum() for o1, o2 in zip(*outs)])
    print("the average run time of keras: ", avg_time[0], "the average run time  of our implementation: ", avg_time[1])
    print('Ratio speed: (our_implementation/keras)', avg_time[1] / avg_time[0])

    # print(A.shape)
    # print("#"*10)
    # print(o1[0,1,1,:])
    # print("#" * 10)
    # print(o2[0,1,1,:])