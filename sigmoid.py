import numpy as np


def sigmoid(x):
    return (1 + np.exp(-x)) ** -1


def sigmoid_derivative(x):
    s = sigmoid(x)
    return s * (1 - s)
