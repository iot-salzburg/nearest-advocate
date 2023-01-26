
import numpy as np
cimport numpy as np
cimport cython


FACTOR = 1e3  # numeric precision of the integer timestamps
    
    
def nearest_advocate_wrapper(arr_ref_np, arr_sig_np, dist_max=0.0, dist_padding=0.0, regulate_paddings=1):
    cdef np.ndarray[np.int64_t, ndim=1] arr_ref = (FACTOR*arr_ref_np).astype(np.int64)
    cdef np.ndarray[np.int64_t, ndim=1] arr_sig = (FACTOR*arr_sig_np).astype(np.int64)
    mean_dist = nearest_advocate_c(arr_ref, arr_sig, 
                                   dist_max=(FACTOR*dist_max).astype(np.int64), 
                                   dist_padding=(FACTOR*dist_padding).astype(np.int64),
                                   regulate_paddings=regulate_paddings)
    return mean_dist / FACTOR


cdef np.float32_t nearest_advocate_c(np.ndarray[np.int64_t, ndim=1] arr_ref, 
                                     np.ndarray[np.int64_t, ndim=1] arr_sig, 
                                     np.int64_t dist_max=0, np.int64_t dist_padding=0, 
                                     np.uint8_t regulate_paddings=1):
    '''Compares the R-peaks of two arrays that are moved by time_delta.
    arr_ref (np.array): Reference array with timestamps assumed to be correct
    arr_sig (np.array): Array that is not synchroneous
    dist_max (None, float): Maximal accepted distances, default None: 1/4 of the median gap of arr_ref
    dist_padding (None, float): Assumed distances of non-overlapping (padding) matches, default None: 1/4 of the median gap of arr_ref
    regulate_paddings (bool): regulate non-overlapping events in arr_sig with a maximum distance of err_max
    ''' 
    cdef np.uint32_t i1 = 0, i2 = 0, counter = 0
    cdef np.int64_t cum_distance = 0
    
    cdef np.uint32_t l_arr_ref = arr_ref.shape[0]
    cdef np.uint32_t l_arr_sig = arr_sig.shape[0]

    # cut leading reference timestamps
    while i1+1 < l_arr_ref and arr_ref[i1+1] <= arr_sig[i2]:
        i1 += 1
    # solve non-overlapping case
    if i1+1 == l_arr_ref:
        # if verbose >= 1:
        #     print("non-overlapping before arr_ref")
        return dist_max
    # if verbose >= 2:
    #     print("Cut off leading ref: \t", arr_ref[i1], arr_sig[i2], arr_ref[i1+1])   
    # assert arr_ref[i1+1] > arr_sig[i2]
    
    # count leading signal timestamps
    while i2 < l_arr_sig and arr_sig[i2] < arr_ref[i1]:
        # if verbose >= 2:
        #     print("Count leading signal: \t", arr_ref[i1], arr_sig[i2], arr_ref[i1+1])  
        if regulate_paddings != 0:
            cum_distance += min(arr_ref[i1]-arr_sig[i2], dist_padding)
            counter += 1
        i2 += 1
    # solve non-overlapping case
    if i2 == l_arr_sig:
        # if verbose >= 1:
        #     print("non-overlapping before arr_sig")
        return dist_max     
    
    # regular cases and trailing reference timestamps
    while i2 < l_arr_sig:
        # regular case
        if arr_sig[i2] < arr_ref[-1]:
            # skip arr_ref forward to regalar case
            while i1+1 < l_arr_ref and arr_ref[i1+1] <= arr_sig[i2]:
                i1 += 1
            if i1+1 >= l_arr_ref: 
                i2 += 1
                continue
            # if verbose >= 2:
            #     print("Count regular cases: \t", arr_ref[i1], ts2, arr_ref[i1+1])
            # assert arr_ref[i1] <= ts2
            # assert ts2 < arr_ref[i1+1]
            
            cum_distance += min(arr_sig[i2]-arr_ref[i1], arr_ref[i1+1]-arr_sig[i2], dist_max) 
            counter += 1
        # trailing reference timestamps
        elif regulate_paddings != 0:  
            # assert arr_ref[i1+1] <= ts2
            # if verbose >= 2:
            #     print("Count trailing ref: \t", ts2, arr_ref[i1+1])
            cum_distance += min(arr_sig[i2]-arr_ref[i1+1], dist_padding) 
            counter += 1
        i2 += 1
    
    # return mean cum_distance
    return <np.float32_t>cum_distance / counter

def sparse_search_time_delta_wrapper_c(arr_ref_np, arr_sig_np, td_min, td_max, 
                                     sparse_factor=1, pps=100,
                                     dist_max=0.0, regulate_paddings=True, dist_padding=0.0):
    
    cdef np.ndarray[np.int64_t, ndim=1] arr_ref = (FACTOR*arr_ref_np).astype(np.int64)
    cdef np.ndarray[np.int64_t, ndim=1] arr_sig = (FACTOR*arr_sig_np).astype(np.int64)
                                                                             
    cdef np.ndarray[np.float32_t, ndim=2] np_nearest = np.empty((int((td_max-td_min)*pps), 2), dtype=np.float32)
    # np_nearest[:, 0] = np.arange(td_min, td_max, 1/pps)
    idx = 0
    td = td_min
    while idx < np_nearest.shape[0]:
        np_nearest[idx, 0] = td
        td += 1/pps
        idx += 1
    
    return sparse_search_time_delta_c(arr_ref, arr_sig, td_min=td_min, td_max=td_max, 
                                      np_nearest=np_nearest, sparse_factor=sparse_factor, pps=pps,
                                      dist_max=FACTOR*dist_max,
                                      dist_padding=FACTOR*dist_padding, 
                                      regulate_paddings=regulate_paddings)

cdef sparse_search_time_delta_c(arr_ref, arr_sig, td_min, td_max, np_nearest, 
                                sparse_factor=1, pps=100,
                             dist_max=0.0, regulate_paddings=True, dist_padding=0.0):
    '''Compares the arrays for a range of time_deltas in a given interval.
    arr_ref (np.array): Reference array with timestamps assumed to be correct
    arr_sig (np.array): Array that is not synchroneous
    td_min (float): lower bound of the search space for the time shift of arr_ref
    td_max (float): upper bound of the search space for the time shift of arr_ref
    sparse_factor (int): factor for the sparseness of arr_sig for the calculation (default 1)
    pps (int): number of investigated time-shifts per second, should be higher than 10 times the number of median gap of arr_ref (default 100).
    dist_max (None, float): Maximal accepted distances, default None: 1/4 of the median gap of arr_ref
    dist_padding (None, float): Assumed distances of non-overlapping (padding) matches, default None: 1/4 of the median gap of arr_ref
    regulate_paddings (bool): regulate non-overlapping events in arr_sig with a maximum distance of err_max
    '''
    # set the default values if unset
    # set dist_max and dist_padding relative to the median gap of the timestamps in the reference array if not set
    # TODO improve default value: min(np.median(np.diff(arr_sig)), np.median(np.diff(arr_ref))) / 4
    if dist_max == 0.0:
        dist_max = min(np.median(np.diff(arr_ref))/4, np.median(np.diff(arr_sig))/4)
    if dist_padding == 0.0:
        dist_padding = min(np.median(np.diff(arr_ref))/4, np.median(np.diff(arr_sig))/4)
        
    # Random subsample
    if sparse_factor > 1:
        arr_sig = arr_sig[sparse_factor//2::sparse_factor]

    idx = 0
    while idx < np_nearest.shape[0]:
        # calculate the nearest advocate criteria of both arrays
        mean_dist = nearest_advocate_c(
            arr_ref, arr_sig - <np.int64_t>(FACTOR*np_nearest[idx,0]), # Arr 2 is shifted by time-delta
            dist_max=<np.int64_t>dist_max, dist_padding=<np.int64_t>dist_padding,
            regulate_paddings=regulate_paddings) 
        np_nearest[idx,1] = <np.float32_t>(mean_dist / FACTOR)
        idx += 1
    return np_nearest
