cimport numpy as np
from numpy.lib.stride_tricks import as_strided
from math import floor

DTYPE = np.float64
ctypedef np.float64_t DTYPE_t

def conv_forward_strides(np.ndarray[DTYPE_t, ndim=4] x, np.ndarray[DTYPE_t, ndim=4] weights, int b, int s, int p):
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

