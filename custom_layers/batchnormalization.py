import numpy as np

if __name__ == "__main__":
    import os
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # to ignore warning errors
    import time
    import tensorflow.keras.backend as keras
    import tensorflow.keras.layers as layers


    # # data_type = channels_last
    A = np.random.randn(2, 2, 2, 2)*10
    avg_time = [0, 0]
    outs = [[], []]
    x = keras.constant(A)
    batchnormalization_keras = layers.BatchNormalization()

    n_simulations = 10
    for _ in range(n_simulations):
        # Keras
        t1 = time.time()
        o1 = batchnormalization_keras(x).numpy()
        avg_time[0] += (time.time() - t1)/n_simulations
        outs[0].append(o1)

        # # our methods
        # t1 = time.time()
        # o2 = zeropadding2d(A, 1)
        # avg_time[1] += (time.time() - t1)/n_simulations
        # outs[1].append(o2)

    # print("difference of predictions ", [(o1 - o2).sum() for o1, o2 in zip(*outs)])
    # print("the average run time of keras: ", avg_time[0], "the average run time  of our implementation: ", avg_time[1])
    # print('Ratio speed: (our_implementation/keras)', avg_time[1] / avg_time[0])

    print(A)
    print("#"*10)
    print(o1)
    # print("#" * 10)
    # print(o2)