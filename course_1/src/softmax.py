import numpy as np


def softmax(x):
    """
    Calculates the softmax for each row of the input x.

    Argument:
    x -- A numpy matrix of shape (m,n)
    """
    # Apply exp() element-wise to x.
    x_exp = np.exp(x)

    # Create a vector x_sum that sums each row of x_exp.
    x_sum = np.sum(x_exp, axis=1, keepdims=True)

    # Get softmax(x) by dividing x_exp by x_sum element-wise - works due to numpy broadcasting.
    s = x_exp / x_sum
    return s
