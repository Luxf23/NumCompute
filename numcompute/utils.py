"""
utils.py

Utility functions for data preprocessing.
"""

import numpy as np


def normalize(data):
    arr = np.asarray(data, dtype=float)

    if arr.size == 0:
        raise ValueError("Input must not be empty.")

    norm = np.linalg.norm(arr)
    if norm == 0:
        return arr

    return arr / norm


def minmax_scale(data):
    arr = np.asarray(data, dtype=float)

    if arr.size == 0:
        raise ValueError("Input must not be empty.")

    min_val = np.min(arr)
    max_val = np.max(arr)

    if max_val == min_val:
        return np.zeros_like(arr)

    return (arr - min_val) / (max_val - min_val)


def clip(data, min_value, max_value):
    arr = np.asarray(data)

    if min_value > max_value:
        raise ValueError("min_value must be <= max_value")

    return np.clip(arr, min_value, max_value)