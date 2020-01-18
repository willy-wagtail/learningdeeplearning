import numpy as np


def sigmoid(value):
    s = (1 + np.exp(-value)) ** -1
    return s


def sigmoid_derivative(value):
    s = sigmoid(value)
    ds = s * (1 - s)
    return ds


x = np.array([1, 2, 3])

print("sigmoid(x) = " + str(sigmoid(x)))

print("sigmoid_derivative(x) = " + str(sigmoid_derivative(x)))
