import numpy as np


def L2(y_predicted, y_actual):
    diff = np.subtract(y_predicted, y_actual)
    loss = np.dot(diff, diff)
    return loss
