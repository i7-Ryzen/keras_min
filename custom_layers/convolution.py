
def conv_forward_strides(x, weights, b, s, p):
    """
    Input:
    - x: [N  * H * W * D] -> N image data of size D * H * W
    - w: [D * F1 * F1 * K] -> K filters of size D * F * F
    - b: [K * 1]   -> bias
    - s: stride of convolution (integer)
    - p: size of zero padding (integer)
    Output:
    - f: [N * K * H_ * W_] -> activation maps, where
      - H_ = (H - F + 2P)/S + 1
      - W_ = (W - F + 2P)/S + 1
    - X_col: [(D * F * F) * (H_ * W_ * N)] -> column stretched images
    """

    # w = w.transpose(2 ,0 ,1 ,3)
    d, f1, f2, k = weights.shape
    n, h, w, d = x.shape
    assert (H - F + 2* p) % s == 0, 'wrong convolution params'
    assert (W - F + 2 * p) % s == 0, 'wrong convolution params'

    H_ = floor((H - F + 2 * p) / s) + 1
    W_ = floor((W - F + 2 * p) / s) + 1
    X_col = im2col_forward(x, F, F, s, p)
    f = np.dot(X_col.transpose(0, 2, 1), w.reshape(-1, K), ) + b
    f = f.reshape(N, H_, W_, K)
    return f, X_col




######################################################################
# test : predictions & run_time
######################################################################
import time
import tensorflow.keras.backend as keras
import tensorflow.keras.layers as layers

A = np.random.randn(1, 200, 200, 1)


avg_time = [0, 0]
outs = [[], []]
x = keras.constant(A)
# pool2d_keras = layers.MaxPooling2D(pool_size=(2, 2), padding="same", input_shape=x.shape[1:])
conv2d_keras = layers.Conv2D(filters=1, kernel_size=(3, 3), strides=(1, 1), padding="same", activation=None,
                             input_shape=x.shape)

# y = pool2d_keras(x).numpy()

#res = pool2d(w, kernel_size=2, stride=2, padding=0, pool_mode='max')

n_simulations = 4
for _ in range(n_simulations):
    # Keras
    t1 = time.time()
    o1 = pool2d_keras(x).numpy()
    avg_time[0] += time.time() - t1
    outs[0].append(o1)

    # # our methods
    # t1 = time.time()
    # o2 = pool2d(w, kernel_size=2, stride=2, padding=0, pool_mode='max')
    # avg_time[1] += time.time() - t1
    # outs[1].append(o2)

# print("difference of predictions ", [(o1 - o2).sum() for o1, o2 in zip(*outs)])
# print("the average run time of keras: ", avg_time[0], "the average run time  of our implementation: ", avg_time[1])
# print('Ratio speed: (our_implementation/keras)', avg_time[1]/avg_time[0])