import numpy as np


def relu(x):
    return np.maximum(x, 0)

def linear(x):
    return x


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def softmax_function(x):
    expo = np.exp(x)
    expo_sum = np.sum(np.exp(x))
    return expo / expo_sum


def softplus_function(x):
    expo = np.exp(x)
    return np.log(expo + 1)


def softsign_function(x):
    absx = np.abs(x) + 1
    return x / absx


def tanh_function(x):
    return np.tanh(x)


def elu_function(x, alpha):
    return alpha * (np.exp(x) - 1) * (x > 0)
