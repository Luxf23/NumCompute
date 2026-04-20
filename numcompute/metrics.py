"""
metrics.py

Basic statistical metrics.
"""

from __future__ import annotations
import numpy as np


def _validate_array(data):
    arr = np.asarray(data)

    if arr.size == 0:
        raise ValueError("Input array must not be empty.")

    return arr


def mean(data):
    arr = _validate_array(data)
    return float(np.sum(arr) / arr.size)


def median(data):
    arr = _validate_array(data)
    arr_sorted = np.sort(arr)

    n = arr.size
    mid = n // 2

    if n % 2 == 1:
        return float(arr_sorted[mid])
    else:
        return float((arr_sorted[mid - 1] + arr_sorted[mid]) / 2)


def variance(data):
    arr = _validate_array(data)
    m = mean(arr)
    return float(np.sum((arr - m) ** 2) / arr.size)


def std(data):
    return float(np.sqrt(variance(data)))


def mse(y_true, y_pred):
    y_true = _validate_array(y_true)
    y_pred = _validate_array(y_pred)

    if y_true.shape != y_pred.shape:
        raise ValueError("Shapes must match.")

    return float(np.mean((y_true - y_pred) ** 2))