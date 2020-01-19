import numpy as np


def normalize_rows(x):
    """
    Normalizes each row of the matrix x (to have unit length) with norm 2 of x.

    Argument:
    x -- A numpy matrix of shape (n, m)
    """

    x_norm = np.linalg.norm(x, axis=1, keepdims=True)
    x = x / x_norm
    return x
