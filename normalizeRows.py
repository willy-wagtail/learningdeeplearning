import numpy as np


def normalizeRows(x):
    """
    Normalizes each row of the matrix x (to have unit length) with norm 2 of x.

    Argument:
    x -- A numpy matrix of shape (n, m)
    """

    x_norm = np.linalg.norm(x, axis=1, keepdims=True)
    x = x / x_norm
    return x


test = np.array([[0, 3, 4], [1, 6, 4]])

print("normalizeRows(x) = " + str(normalizeRows(test)))
