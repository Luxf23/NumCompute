"""
optim.py

Basic optimization utilities.
"""

import numpy as np


def gradient_descent_step(x, grad, lr=0.01):
    """
    Perform one step of gradient descent.
    """
    x = np.asarray(x, dtype=float)
    grad = np.asarray(grad, dtype=float)

    if x.shape != grad.shape:
        raise ValueError("Shapes must match.")

    return x - lr * grad