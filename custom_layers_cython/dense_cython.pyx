import numpy as np
cimport numpy as np



DTYPE = np.float64
ctypedef np.float64_t dtype_t


def dense_layer_cython(np.ndarray[dtype_t, ndim=4] x, np.ndarray[dtype_t, ndim=2] w, np.ndarray[dtype_t, ndim=1] b):

  cdef  out = np.dot(x, w) + b
  return out
