#!/usr/bin/env python
"""Tester for the nearest_advocate algorithm for a single time-shift. Comparision of runtime for Numba and Cython."""

import time
import numpy as np
np.random.seed(0)

from numba import njit
from src import nearest_advocate_nb #import nearest_advocate_single


N = 100_000               # number of events in the random arrays
TIME_SHIFT = np.pi        # true time-shift between the two arrays
DEF_DIST = 0.25           # default values for dist_max and dist_padding of nearest_advocate
REGULATE_PADDINGS = True  # regulate the paddings in nearest_advocate


# Create two related event-based array, they differ by a time-shift and gaussian noise
arr_ref = np.sort(np.cumsum(np.random.normal(loc=1, scale=0.25, size=N))).astype(np.float32)
arr_sig = np.sort(arr_ref + TIME_SHIFT + np.random.normal(loc=0, scale=0.1, size=N)).astype(np.float32)

# Time the numba-solution
mean_dist = nearest_advocate_single(arr_ref=arr_ref, arr_sig=arr_sig, 
                                    dist_max=DEF_DIST, dist_padding=DEF_DIST,
                                    regulate_paddings=REGULATE_PADDINGS)
start_time = time.time()
mean_dist = nearest_advocate_single(arr_ref=arr_ref, arr_sig=arr_sig, 
                                    dist_max=DEF_DIST, dist_padding=DEF_DIST, 
                                    regulate_paddings=REGULATE_PADDINGS)
pytime = time.time() - start_time
print(f"Numba:   \t{pytime:.8f} s, \tmean distance: {mean_dist:.6f} s")


# Time the Cython-solution
from src.nearest_advocate_c import nearest_advocate_single
start_time = time.time()
mean_dist = nearest_advocate_single(arr_ref=arr_ref, arr_sig=arr_sig, 
                                      dist_max=DEF_DIST, dist_padding=DEF_DIST, 
                                      regulate_paddings=REGULATE_PADDINGS)
pytime = time.time() - start_time
print(f"Cython: \t{pytime:.8f} s, \tmean distance: {mean_dist:.6f} s")
