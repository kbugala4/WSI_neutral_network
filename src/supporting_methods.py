import numpy as np


def ReLu(x, is_deriv=False):
    if is_deriv:
        return 1 * (x >= 0)
    else:
        return x * (x >= 0)


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def softmax(x):
    print((np.exp(x) / np.sum(np.exp(x), axis=0)).shape)
    return np.exp(x) / np.sum(np.exp(x), axis=0)
