import numpy as np


def L1(y_predicted, y_actual):
    loss = np.sum(np.abs(y_predicted - y_actual))
    return loss


y_hat = np.array([.9, 0.2, 0.1, .4, .9])
y = np.array([1, 0, 0, 1, 1])

print("L1 = " + str(L1(y_hat, y)))
