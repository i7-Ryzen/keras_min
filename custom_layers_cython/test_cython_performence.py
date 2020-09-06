
# first compile the .pyx code with : python setup.py build_ext --inplace

######################################################################
# test : predictions & run_time
######################################################################
if __name__ == "__main__":
    from dense_cython import dense_layer_cython
    import os
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # to ignore warning errors
    import time
    import tensorflow.keras.backend as keras
    import tensorflow.keras.layers as layers
    import numpy as np


    A = np.random.randn(3, 200, 200, 3)
    #A= np.asarray(A,dtype=np.float64)
    avg_time = [0, 0]
    outs = [[], []]
    x = keras.constant(A)
    dense_keras = layers.Dense(units=1024, activation=None, input_shape=x.shape)
    y = dense_keras(x).numpy()
    W, b = dense_keras.get_weights()
    W = np.asarray(W, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)




    n_simulations = 3
    for _ in range(n_simulations):
        # Keras
        t1 = time.time()
        o1 = dense_keras(x).numpy()
        avg_time[0] += (time.time() - t1)/n_simulations
        outs[0].append(o1)

        # our methods
        t1 = time.time()
        o2 = dense_layer_cython(A, W, b)
        avg_time[1] += (time.time() - t1)/n_simulations
        outs[1].append(o2)


    print("difference of predictions ", [(o1 - o2).sum() for o1, o2 in zip(*outs)])
    print("the average run time of keras: ", avg_time[0], "the average run time  of cython implementation: ", avg_time[1])
    print('Ratio speed: (our_implementation/keras)', avg_time[1]/avg_time[0])