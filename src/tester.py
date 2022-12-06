import time
import numpy as np
from numba import njit
from nearest_advocate import nearest_advocate
from nearest_advocate_c import nearest_advocate_wrapper


N = 100_000
TIME_SHIFT = np.pi
DEF_DIST = 0.25
REGULATE_PADDINGS = True

# np.random.seed(0)
arr_ref = np.sort(np.cumsum(np.random.normal(loc=1, scale=0.25, size=N))).astype(np.float32)
arr_sig = np.sort(arr_ref + TIME_SHIFT + np.random.normal(loc=0, scale=0.1, size=N)).astype(np.float32)

# @njit
# def fast_median_diff(arr):
#     periodicy = 1
#     if arr.shape[0] > 1000:
#         periodicy = int(np.ceil(arr.shape[0] / 1000))
#     i = 0
#     si = 0
#     short_arr = np.zeros(1000, dtype=np.float32)
#     while i < arr.shape[0]-1:
#         if i % periodicy == 0:
#             short_arr[si] = arr[i+1] - arr[i]
#             si += 1
#         i += 1
#     print(si)
#     return np.median(short_arr[:si])

# for ne in range(1, 8):
#     N = 10**ne
#     arr_ref = np.sort(np.cumsum(np.random.normal(loc=1, scale=0.25, size=N))).astype(np.float32)
    
#     start_time = time.time()
#     median = np.median(np.diff(arr_ref))
#     median_time = time.time() - start_time
    
#     start_time = time.time()
#     median_fast = fast_median_diff(arr_ref)
#     fast_time = time.time() - start_time
    
#     start_time = time.time()
#     _ = np.sort(np.diff(arr_ref))
#     sort_time = time.time() - start_time
    
#     print(f"N={N}: \tMedian: {median_time:.10f}s ({median:.4f}),\t {fast_time:.10f}s ({median_fast:.4f}),\t {sort_time:.10f}s")



# Start the test
start_time = time.time()
mean_dist = nearest_advocate(arr_ref, arr_sig, 
                             dist_max=DEF_DIST, dist_padding=DEF_DIST, 
                             regulate_paddings=REGULATE_PADDINGS)
pytime = time.time() - start_time
print(f"Python: \t{pytime:.8f} s, \tmean_dist: {mean_dist:.6f} s")

start_time = time.time()
mean_dist = nearest_advocate_wrapper(arr_ref, arr_sig, 
                             dist_max=DEF_DIST, dist_padding=DEF_DIST, 
                                     regulate_paddings=REGULATE_PADDINGS)
pytime = time.time() - start_time
print(f"Cython: \t{pytime:.8f} s, \tmean_dist: {mean_dist:.6f} s")

nearest_advocate_nb = njit(nearest_advocate)
mean_dist = nearest_advocate_nb(arr_ref, arr_sig, 
                             dist_max=DEF_DIST, dist_padding=DEF_DIST, 
                                regulate_paddings=REGULATE_PADDINGS)
start_time = time.time()
mean_dist = nearest_advocate_nb(arr_ref, arr_sig, 
                             dist_max=DEF_DIST, dist_padding=DEF_DIST, 
                                regulate_paddings=REGULATE_PADDINGS)
pytime = time.time() - start_time
print(f"Numba:   \t{pytime:.8f} s, \tmean_dist: {mean_dist:.6f} s")

