import numpy as np
import pytest

from numcompute.metrics import mean, median, variance, std, mse


def test_mean():
    arr = np.array([1, 2, 3, 4])
    assert mean(arr) == 2.5


def test_median_even():
    arr = np.array([1, 2, 3, 4])
    assert median(arr) == 2.5


def test_median_odd():
    arr = np.array([1, 2, 3])
    assert median(arr) == 2.0


def test_variance():
    arr = np.array([1, 2, 3])
    assert np.isclose(variance(arr), 2/3)


def test_std():
    arr = np.array([1, 2, 3])
    assert np.isclose(std(arr), np.sqrt(2/3))


def test_mse():
    y1 = np.array([1, 2, 3])
    y2 = np.array([1, 2, 4])
    assert mse(y1, y2) == 1/3


def test_empty_input():
    with pytest.raises(ValueError):
        mean([])