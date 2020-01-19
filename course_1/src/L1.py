import numpy as np


def L1(y_predicted, y_actual):
    loss = np.sum(np.abs(y_predicted - y_actual))
    return loss
