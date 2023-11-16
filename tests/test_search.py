#!/usr/bin/env python
"""Tester for the nearest_advocate algorithm for a given search space of time-shifts. Comparision of runtime for Numba and Cython."""

import sys
import time
import numpy as np
np.random.seed(0)

from numba import njit

sys.path.append("src")


N = 10_000                # number of events in the random arrays
TIME_SHIFT = np.pi        # true time-shift between the two arrays
DEF_DIST = 0.25           # default values for dist_max of nearest_advocate
TD_MAX = 60               # search space of +-1 minutes
TD_MIN = -60
SAMPLES_PER_S = 20        # precision of the search space


# Create two related event-based array, they differ by a time-shift and gaussian noise
arr_ref = np.sort(np.cumsum(np.random.normal(loc=1, scale=0.25, size=N))).astype(np.float32)
arr_sig = np.sort(arr_ref + TIME_SHIFT + np.random.normal(loc=0, scale=0.1, size=N)).astype(np.float32)

# Time the numba-solution
from nearest_advocate_numba import nearest_advocate
# run once before the test to just-in-time compile it
np_nearest = nearest_advocate(arr_ref=arr_ref, arr_sig=arr_sig,
                              td_min=-1, td_max=1, sps=SAMPLES_PER_S, sparse_factor=1,
                              dist_max=DEF_DIST)

start_time = time.time()
np_nearest = nearest_advocate(arr_ref=arr_ref, arr_sig=arr_sig,
                              td_min=TD_MIN, td_max=TD_MAX, sps=SAMPLES_PER_S, sparse_factor=1,
                              dist_max=DEF_DIST)
pytime = time.time() - start_time
time_shift, min_mean_dist = np_nearest[np.argmin(np_nearest[:,1])]
print(f"Numba:   \t{pytime:.8f} s, \t detected time shift: {time_shift:.2f} s, \t minimal mean distance: {min_mean_dist:.6f} s")


# # Time the Cython-solution
# from nearest_advocate_cython import nearest_advocate
# start_time = time.time()
# np_nearest = nearest_advocate(arr_ref=arr_ref, arr_sig=arr_sig,
#                                 td_min=TD_MIN, td_max=TD_MAX, sps=SAMPLES_PER_S, sparse_factor=1,
#                                 dist_max=DEF_DIST)
# pytime = time.time() - start_time
# time_shift, min_mean_dist = np_nearest[np.argmin(np_nearest[:,1])]
# print(f"Cython:   \t{pytime:.8f} s, \t detected time shift: {time_shift:.2f} s, \t minimal mean distance: {min_mean_dist:.6f} s")
