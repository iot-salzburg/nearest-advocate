#!/usr/bin/env python
"""Tester for the nearest_advocate algorithm for a given search space of time-shifts, validates Numba, Cython and Python."""

import numpy as np
from numpy.testing import assert_equal, assert_almost_equal

import sys
sys.path.append("src")

SEED = 0


def test_nearest_advocate_base():
    N = 1_000
    DEF_DIST = 0.25

    np.random.seed(SEED)
    arr_ref = np.sort(np.cumsum(np.random.normal(
        loc=1, scale=0.25, size=N))).astype(np.float32)
    arr_sig = np.sort(arr_ref + np.pi + np.random.normal(
        loc=0, scale=0.1, size=N)).astype(np.float32)

    np_nearest = nearest_advocate(
        arr_ref=arr_ref, arr_sig=arr_sig,
        td_min=-60, td_max=60, sps=20, sparse_factor=1,
        dist_max=DEF_DIST, regulate_paddings=True, dist_padding=DEF_DIST)
    time_shift, min_mean_dist = np_nearest[np.argmin(np_nearest[:, 1])]

    assert_almost_equal(time_shift, np.pi, decimal=1)
    assert_almost_equal(min_mean_dist, 0.07694374, decimal=2)


def test_nearest_advocate_edge():
    N = 1_000
    DEF_DIST = 0.25

    np.random.seed(SEED)
    arr_ref = np.sort(np.cumsum(np.random.normal(
        loc=1, scale=0.5, size=N))).astype(np.float32)
    arr_sig = np.sort(arr_ref + np.pi + np.random.normal(
        loc=0, scale=0.4, size=N)).astype(np.float32)

    np_nearest = nearest_advocate(
        arr_ref=arr_ref, arr_sig=arr_sig,
        td_min=-60, td_max=60, sps=20, sparse_factor=1,
        dist_max=DEF_DIST, regulate_paddings=True, dist_padding=DEF_DIST)
    time_shift, min_mean_dist = np_nearest[np.argmin(np_nearest[:, 1])]

    assert_almost_equal(time_shift, np.pi, decimal=1)
    assert_almost_equal(min_mean_dist, 0.1646458, decimal=2)


def test_nearest_advocate_base_defmax():
    N = 1_000
    DEF_DIST = 0.0

    np.random.seed(SEED)
    arr_ref = np.sort(np.cumsum(np.random.normal(
        loc=1, scale=0.25, size=N))).astype(np.float32)
    arr_sig = np.sort(arr_ref + np.pi + np.random.normal(
        loc=0, scale=0.1, size=N)).astype(np.float32)

    np_nearest = nearest_advocate(
        arr_ref=arr_ref, arr_sig=arr_sig,
        td_min=-60, td_max=60, sps=20, sparse_factor=1,
        dist_max=DEF_DIST, regulate_paddings=True, dist_padding=DEF_DIST)
    time_shift, min_mean_dist = np_nearest[np.argmin(np_nearest[:, 1])]

    assert_almost_equal(time_shift, np.pi, decimal=1)
    assert_almost_equal(min_mean_dist, 0.07690712, decimal=2)


def test_nearest_advocate_base_fewoverlap():
    N = 1_000
    DEF_DIST = 0.0
    TIME_SHIFT = 900

    np.random.seed(SEED)
    arr_ref = np.sort(np.cumsum(np.random.normal(
        loc=1, scale=0.25, size=N))).astype(np.float32)
    arr_sig = np.sort(arr_ref + TIME_SHIFT + np.random.normal(
        loc=0, scale=0.1, size=N)).astype(np.float32)

    np_nearest = nearest_advocate(
        arr_ref=arr_ref, arr_sig=arr_sig,
        td_min=850, td_max=950, sps=20, sparse_factor=1,
        dist_max=DEF_DIST, regulate_paddings=True, dist_padding=DEF_DIST)
    time_shift, min_mean_dist = np_nearest[np.argmin(np_nearest[:, 1])]

    assert_almost_equal(time_shift, TIME_SHIFT, decimal=1)
    assert_almost_equal(min_mean_dist, 0.07674612, decimal=2)


def test_nearest_advocate_base_nopadding():
    N = 1_000
    DEF_DIST = 0.25

    np.random.seed(SEED)
    arr_ref = np.sort(np.cumsum(np.random.normal(
        loc=1, scale=0.25, size=N))).astype(np.float32)
    arr_sig = np.sort(arr_ref + np.pi + np.random.normal(
        loc=0, scale=0.1, size=N)).astype(np.float32)

    np_nearest = nearest_advocate(
        arr_ref=arr_ref, arr_sig=arr_sig,
        td_min=-60, td_max=60, sps=20, sparse_factor=1,
        dist_max=DEF_DIST, regulate_paddings=False)
    time_shift, min_mean_dist = np_nearest[np.argmin(np_nearest[:, 1])]

    assert_almost_equal(time_shift, np.pi, decimal=1)
    assert_almost_equal(min_mean_dist, 0.07690712, decimal=2)


def test_nearest_advocate_base_noverlap():
    N = 100
    DEF_DIST = 0.25
    TIME_SHIFT = 200

    np.random.seed(SEED)
    arr_ref = np.sort(np.cumsum(np.random.normal(
        loc=1, scale=0.25, size=N))).astype(np.float32)
    arr_sig = np.sort(arr_ref + TIME_SHIFT + np.random.normal(
        loc=0, scale=0.1, size=N)).astype(np.float32)

    np_nearest = nearest_advocate(
        arr_ref=arr_ref, arr_sig=arr_sig,
        td_min=-60, td_max=60, sps=20, sparse_factor=1,
        dist_max=DEF_DIST, regulate_paddings=False)
    time_shift, min_mean_dist = np_nearest[np.argmin(np_nearest[:, 1])]

    assert_almost_equal(time_shift, -60, decimal=1)  # each value is the same
    assert_equal(min_mean_dist, DEF_DIST)


def test_nearest_advocate_bigtime():
    N = 1_000
    DEF_DIST = 0.25
    time_unix = 1692720007.811548

    np.random.seed(SEED)
    arr_ref = np.sort(np.cumsum(np.random.normal(
        loc=1, scale=0.25, size=N)) + time_unix)
    arr_sig = np.sort(arr_ref + np.pi + np.random.normal(
        loc=0, scale=0.1, size=N))

    # subtract the minimal event timestamp
    min_event_time = min(arr_ref[0], arr_sig[0])
    arr_ref -= min_event_time
    arr_sig -= min_event_time

    np_nearest = nearest_advocate(
        arr_ref=arr_ref, arr_sig=arr_sig,
        td_min=-60, td_max=60, sps=20, sparse_factor=1,
        dist_max=DEF_DIST, regulate_paddings=True, dist_padding=DEF_DIST)
    time_shift, min_mean_dist = np_nearest[np.argmin(np_nearest[:, 1])]

    assert_almost_equal(time_shift, np.pi, decimal=1)
    assert_almost_equal(min_mean_dist, 0.07694374, decimal=2)


if __name__ == "__main__":
    print("Testing generic-version:  \t", end="")
    from nearest_advocate import nearest_advocate
    test_nearest_advocate_base()
    test_nearest_advocate_edge()
    test_nearest_advocate_base_defmax()
    test_nearest_advocate_base_fewoverlap()
    test_nearest_advocate_base_nopadding()
    test_nearest_advocate_base_noverlap()
    test_nearest_advocate_bigtime()
    print("ok")

    print("Testing numba-version:  \t", end="")
    from nearest_advocate_numba import nearest_advocate
    test_nearest_advocate_base()
    test_nearest_advocate_edge()
    test_nearest_advocate_base_defmax()
    test_nearest_advocate_base_fewoverlap()
    test_nearest_advocate_base_nopadding()
    test_nearest_advocate_base_noverlap()
    # test_nearest_advocate_bigtime()  # only for caller
    print("ok")

    # print("Testing Cython-version:  \t", end="")
    # from nearest_advocate_cython import nearest_advocate
    # test_nearest_advocate_base()
    # test_nearest_advocate_edge()
    # test_nearest_advocate_base_defmax()
    # test_nearest_advocate_base_fewoverlap()
    # test_nearest_advocate_base_nopadding()
    # test_nearest_advocate_base_noverlap()
    # print("ok")

    print("Testing Python-version:  \t", end="")
    from nearest_advocate_python import nearest_advocate
    test_nearest_advocate_base()
    test_nearest_advocate_edge()
    test_nearest_advocate_base_defmax()
    test_nearest_advocate_base_fewoverlap()
    test_nearest_advocate_base_nopadding()
    test_nearest_advocate_base_noverlap()
    # test_nearest_advocate_bigtime()  # only for caller
    print("ok")
