import numpy as np


def flatten_layer(x):
    """
  flatten the layer
  Inputs:
  x - Input data, of shape (N, d_1, ..., d_k)

  Returns a tuple of:
  - out: output, of shape (N, M)
  - cache: x
  """
    out = x.reshape([1, -1])
    return out


def dense_layer(x, w, b):
    """
  Computes the forward pass for an affine (fully-connected) layer.
  The input x has shape (N, D) where x[i] is the ith input.
  We multiply this against a weight matrix of shape (D, M)

  Inputs:
  x - Input data, of shape (N, D)
  w - Weights, of shape (D, M)
  b - Biases, of shape (M,)

  Returns a tuple of:
  - out: output, of shape (N, M)
  - cache: (x, w, b)
  """
    out =  np.matmul(x, w) + b
    # out = np.dot(x, w) + b
    return out


######################################################################
# test : predictions & run_time
######################################################################

if __name__ == "__main__":
    import os
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # to ignore warning errors
    import time
    import tensorflow.keras.backend as keras
    import tensorflow.keras.layers as layers
    import numpy as np
    from tensorflow.keras.models import Sequential

    A = np.random.randn(1, 200, 200, 3).astype('float32')

    avg_time = [0, 0]
    outs = [[], []]

    n_simulations = 3
    for _ in range(n_simulations):
        # Keras
        dense_keras = layers.Dense(units = 1024, activation=None, input_shape=(1, 200, 200, 3))
        t1 = time.time()
        o1 = dense_keras(A)
        avg_time[0] += (time.time() - t1) / n_simulations
        o1 = o1.numpy()
        outs[0].append(o1)

        # numpy method
        W, b = dense_keras.get_weights()
        t1 = time.time()
        # o2 = np.dot(A, W) + b
        # o2 = np.matmul(A, W) + b
        o2 = dense_layer(A, W, b)
        avg_time[1] += (time.time() - t1) / n_simulations
        outs[1].append(o2)

    print("difference of predictions ", [(o1 - o2).sum() for o1, o2 in zip(*outs)])
    print("the average run time of keras: ", avg_time[0], "the average run time  of our implementation: ", avg_time[1])
    print('Ratio speed: (our_implementation/keras)', avg_time[1] / avg_time[0])

#################################################
# result
#################################################

# difference of predictions  [-2.4116001e-05, -2.4116001e-05, -2.4116001e-05]
# the average run time of keras:  0.2522510687510172 the average run time  of our implementation:  2.8951672712961836
# Ratio speed: (our_implementation/keras) 11.477324102653613

