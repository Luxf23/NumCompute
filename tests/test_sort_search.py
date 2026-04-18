import numpy as np
import pytest

from numcompute.sort_search import (
    stable_sort,
    multi_key_sort,
    topk,
    quickselect,
    binary_search,
)


# =====================
# stable_sort
# =====================
def test_stable_sort_basic():
    arr = np.array([3, 1, 2])
    result = stable_sort(arr)
    assert np.array_equal(result, np.array([1, 2, 3]))


def test_stable_sort_descending():
    arr = np.array([3, 1, 2])
    result = stable_sort(arr, ascending=False)
    assert np.array_equal(result, np.array([3, 2, 1]))


# =====================
# multi_key_sort
# =====================
def test_multi_key_sort_basic():
    arr = np.array([
        [2, 30],
        [1, 20],
        [2, 10],
    ])
    result = multi_key_sort(arr, keys=[0, 1], ascending=[True, True])
    expected = np.array([
        [1, 20],
        [2, 10],
        [2, 30],
    ])
    assert np.array_equal(result, expected)


# =====================
# topk
# =====================
def test_topk_largest():
    arr = np.array([5, 1, 9, 3, 7])
    values, indices = topk(arr, 2)
    assert set(values) == {9, 7}


def test_topk_smallest():
    arr = np.array([5, 1, 9, 3, 7])
    values, indices = topk(arr, 2, largest=False)
    assert set(values) == {1, 3}


# =====================
# quickselect
# =====================
def test_quickselect_smallest():
    arr = np.array([5, 1, 9, 3, 7])
    result = quickselect(arr, 2)
    assert result == 5  # 第3小


def test_quickselect_largest():
    arr = np.array([5, 1, 9, 3, 7])
    result = quickselect(arr, 1, largest=True)
    assert result == 7  # 第2大


# =====================
# binary_search
# =====================
def test_binary_search_found():
    arr = np.array([1, 3, 5, 7, 9])
    idx, found = binary_search(arr, 5)
    assert found is True
    assert idx == 2


def test_binary_search_not_found():
    arr = np.array([1, 3, 5, 7, 9])
    idx, found = binary_search(arr, 4)
    assert found is False
    assert idx == 2