import numpy as np
from numpy.lib.stride_tricks import as_strided
from math import floor

def conv_forward_strides(x, weights, b, s, p):
    """
    Input:
    - x: [n  * h * w * c] -> N image data of size c * H * W
    - weights: [c * f1 * f2 * k] -> k filters of size c * f1 * f2
    - b: [k * 1]   -> bias
    - s: stride of convolution (integer)
    - p: size of zero padding (integer)
    Output:
    - f: [n * k * h_ * w_] -> activation maps, where
      - h_ = (h - f1 + 2p)/s + 1
      - w_ = (w - f2 + 2p)/s + 1
    - x_col: [(d * f1 * f2) * (h_ * w_ * n)] -> column stretched images
    """

    # w = w.transpose(2 ,0 ,1 ,3)
    f1, f2, c, k = weights.shape
    n, h, w, d = x.shape

    assert (h - f1 + 2 * p) % s == 0, 'wrong convolution params'
    assert (w - f2 + 2 * p) % s == 0, 'wrong convolution params'

    h_ = floor((h - f1 + 2 * p)/s) + 1
    w_ = floor((w - f2 + 2 * p)/s) + 1

    # Padding
    # x = np.pad(x, p, mode='constant')
    x = np.pad(x, ((0, 0), (p, p), (p, p), (0, 0)), 'constant')
    # Window view of a
    output_shape = (
        x.shape[0],
        (x.shape[1] - f1) // s + 1,
        (x.shape[2] - f2) // s + 1,
        x.shape[3]
    )
    kernel_size = (f1, f2)
    a_w = as_strided(x,
                     shape=output_shape + kernel_size,
                     strides=(x.strides[0], s * x.strides[1], s * x.strides[2], x.strides[3]) + x.strides[1:3]
                     ).reshape(-1, d*f1*f2)
    weights = np.moveaxis(weights, 2, 0).reshape(-1,k)
    f = np.dot(a_w, weights, ) + b
    return f.reshape(n, h_, w_, k)


######################################################################
# test : predictions & run_time
######################################################################
if __name__ ==  "__main__":
    import os
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # to ignore warning errors
    import time
    import tensorflow.keras.backend as keras
    import tensorflow.keras.layers as layers

    A = np.random.randn(3, 200, 200, 32)

    avg_time = [0, 0]
    outs = [[], []]
    x = keras.constant(A)
    # pool2d_keras = layers.MaxPooling2D(pool_size=(2, 2), padding="same", input_shape=x.shape[1:])
    conv2d_keras = layers.Conv2D(filters=10, kernel_size=(3, 3), strides=(1, 1), padding="same", activation=None,
                                 input_shape=x.shape)

    y = conv2d_keras(x).numpy()
    W, b = conv2d_keras.get_weights()

    n_simulations = 10
    for _ in range(n_simulations):
        # Keras
        t1 = time.time()
        o1 = conv2d_keras(x).numpy()
        # print(o1)
        avg_time[0] += (time.time() - t1)/n_simulations
        outs[0].append(o1)

        # our methods
        t1 = time.time()
        o2 = conv_forward_strides(x=A, weights=W, b=b, s=1, p=1)
        # print(o2)
        avg_time[1] += (time.time() - t1)/n_simulations
        outs[1].append(o2)

    print("difference of predictions ", [(o1 - o2).sum() for o1, o2 in zip(*outs)])
    print("the average run time of keras: ", avg_time[0], "the average run time  of our implementation: ", avg_time[1])
    print('Ratio speed: (our_implementation/keras)', avg_time[1]/avg_time[0])


#################################################
# result
#################################################
# difference of predictions  [-0.00033738004999141387, -0.00033738004999141387, -0.00033738004999141387, -0.00033738004999141387, -0.00033738004999141387, -0.00033738004999141387, -0.00033738004999141387, -0.00033738004999141387, -0.00033738004999141387, -0.00033738004999141387]
# the average run time of keras:  0.012895727157592775 the average run time  of our implementation:  0.11431190967559816
# Ratio speed: (our_implementation/keras) 8.864324460237462
