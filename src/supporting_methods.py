import numpy as np


def ReLu(x, is_deriv=False):
    if is_deriv:
        return 1 * (x >= 0)
    else:
        return x * (x >= 0)


def sigmoid(x, is_deriv=False):
    f = 1.0 / (1.0 + np.exp(-x))
    if is_deriv:
        return f * (f - 1)
    else:
        return f


def softmax(x):
    # print("kebab na cienkim")
    # # print(x)
    # # print(np.exp(x))
    # res = np.exp(x) / np.sum(np.exp(x), axis=0)
    # print(x.shape)
    # print(res.shape)
    # print(np.sum(res, axis=0))
    return np.exp(x) / np.sum(np.exp(x), axis=0)
