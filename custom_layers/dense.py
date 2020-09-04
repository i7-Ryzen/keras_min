# conda install nomkl  (for mac)

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
  out = None
  out = np.dot(x, w) + b
  return out



######################################################################
# test : predictions & run_time
######################################################################

import time
import tensorflow.keras.backend as keras
import tensorflow.keras.layers as layers


A = np.random.randn(3, 200, 200, 3)

avg_time = [0, 0]
outs = [[], []]
x = keras.constant(A)
dense_keras = layers.Dense(units=1024, activation=None, input_shape=x.shape)
y = dense_keras(x).numpy()
W, b = dense_keras.get_weights()

# W = W.transpose(3, 2, 0, 1)


#res = pool2d(w, kernel_size=2, stride=2, padding=0, pool_mode='max')

n_simulations = 4
for _ in range(n_simulations):
    # Keras
    t1 = time.time()
    o1 = dense_keras(x).numpy()
    avg_time[0] += time.time() - t1
    outs[0].append(o1)

    # our methods
    t1 = time.time()
    o2 = dense_layer(A, W, b)
    avg_time[1] += time.time() - t1
    outs[1].append(o2)

print("difference of predictions ", [(o1 - o2).sum() for o1, o2 in zip(*outs)])
print("the average run time of keras: ", avg_time[0], "the average run time  of our implementation: ", avg_time[1])
print('Ratio speed: (our_implementation/keras)', avg_time[1]/avg_time[0])


