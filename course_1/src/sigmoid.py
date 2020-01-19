import numpy as np


def sigmoid(value):
    s = (1 + np.exp(-value)) ** -1
    return s


def sigmoid_derivative(value):
    s = sigmoid(value)
    ds = s * (1 - s)
    return ds
