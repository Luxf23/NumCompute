import numpy as np
import pytest

from numcompute.utils import normalize, minmax_scale, clip


def test_normalize():
    arr = np.array([3, 4])
    result = normalize(arr)
    assert np.allclose(result, np.array([0.6, 0.8]))


def test_minmax_scale():
    arr = np.array([1, 2, 3])
    result = minmax_scale(arr)
    assert np.allclose(result, np.array([0.0, 0.5, 1.0]))


def test_minmax_same_values():
    arr = np.array([5, 5, 5])
    result = minmax_scale(arr)
    assert np.all(result == 0)


def test_clip():
    arr = np.array([1, 5, 10])
    result = clip(arr, 2, 8)
    assert np.array_equal(result, np.array([2, 5, 8]))


def test_clip_invalid():
    with pytest.raises(ValueError):
        clip([1, 2, 3], 5, 2)