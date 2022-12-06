import numpy as np
from numba import njit


# @njit(parallel=False)
def nearest_advocate(arr_ref, arr_sig, dist_max=0.0, dist_padding=0.0, regulate_paddings=True):
    '''Compares the R-peaks of two arrays that are moved by time_delta.
    arr_ref (np.array): Reference array with timestamps assumed to be correct
    arr_sig (np.array): Array that is not synchroneous
    dist_max (None, float): Maximal accepted distances, default None: 1/4 of the median gap of arr_ref
    dist_padding (None, float): Assumed distances of non-overlapping (padding) matches, default None: 1/4 of the median gap of arr_ref
    regulate_paddings (bool): regulate non-overlapping events in arr_sig with a maximum distance of err_max
    ''' 
    i1 = 0
    i2 = 0
    counter = 0
    cum_distance = 0.0
    
    # store the lenghts of the arrays
    l_arr_ref = len(arr_ref)
    l_arr_sig = len(arr_sig)
    
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
        if regulate_paddings:
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
            #     print("Count regular cases: \t", arr_ref[i1], arr_sig[i2], arr_ref[i1+1])
            # assert arr_ref[i1] <= arr_sig[i2]
            # assert arr_sig[i2] < arr_ref[i1+1]
            
            cum_distance += min(arr_sig[i2]-arr_ref[i1], arr_ref[i1+1]-arr_sig[i2], dist_max) 
            counter += 1
        # trailing reference timestamps
        elif regulate_paddings:  
            # assert arr_ref[i1+1] <= arr_sig[i2]
            # if verbose >= 2:
            #     print("Count trailing ref: \t", arr_sig[i2], arr_ref[i1+1])
            cum_distance += min(arr_sig[i2]-arr_ref[i1+1], dist_padding) 
            counter += 1
        i2 += 1
    
    # return mean cum_distance
    return cum_distance / counter
    return cum_distance / counter


# @njit(parallel=True)
# def nearest_advocate(arr_1, arr_2, time_delta, size=1000, dist_max=0.2, dist_padding=None, regulate_paddings=None, verbose=0):
#     '''Compares the R-peaks of two arrays that are moved by time_delta.
#     arr_1 (np.array): Array that is known to be synchroneous
#     arr_2 (np.array): Array that may not be synchroneous
#     time_delta (float): constant deviation of arr_2
#     size (int): Maximal number of R-peaks that are compared. Very high ones are not performant
#     err_out (None, float): If given, the algorithm breaks if this error is surpassed
#     verbose (int): The higher the more output is printed
#     '''
#     # return if the slices don't match
#     if len(arr_1) < 10 or len(arr_2) < 10:
#         return dist_max
    
#     # array 2 is compared with a time delay of time_delta
#     arr_2 = arr_2 - time_delta
    
#     i1 = 0
#     i2 = 0
#     counter = 0
#     err = 0.0
#     arr_1_l = len(arr_1)
    
#     # forward the indices until the intervals for two subsequent peaks intersect. 2 Cases:
#     while i2+1 < len(arr_2) and arr_2[i2] < arr_1[i1]:
#         i2 += 1
        
#     for ts_2 in arr_2[i2:]:
#         # go forward in arr_1 until arr_1[i1] > arr_2[i2]
#         while arr_1[i1+1] <= ts_2 and i1 + 2 < arr_1_l:
#             i1 += 1
#         if i1 + 2 >= arr_1_l:
#             break
            
#         # check if the intervals of subsequent peaks really intersect
#         if not (arr_1[i1] <= ts_2 and ts_2 < arr_1[i1+1]):
#             return dist_max

#         # view two R-peaks for each array
#         err += min(ts_2-arr_1[i1], arr_1[i1+1]-ts_2, dist_max)
#         counter += 1
        
#         # if the error is higher than the limit times number of opposites plus an offset
#         if dist_max and err > dist_max * counter + 0.1:
#             return dist_max
#     if counter > 0:
#         return err / counter
#     else:
#         return dist_max

@njit(parallel=False)
def sparse_search_time_delta(arr_1, arr_2, td_min, td_max, sparse_factor=1, pps=100,
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
        
    np_nearest = np.zeros((int((td_max-td_min)*pps), 2), dtype=np.float32)
    np_nearest[:, 0] = np.arange(td_min, td_max, 1/pps)
    
    # Random subsample
    if sparse_factor > 1:
        probe = arr_2.copy()[sparse_factor//2::sparse_factor]
    else:
        probe = arr_2.copy()
    # probe = arr_2
    for idx in range(np_nearest.shape[0]):
        # calculate the nearest advocate criteria of both arrays
        np_nearest[idx,1] = nearest_advocate(
            arr_1, 
            probe-np_nearest[idx,0],  # Arr 2 is shifted by time-delta
            dist_max=dist_max, regulate_paddings=regulate_paddings, dist_padding=dist_padding)
    return np_nearest


def fast_median(arr):
    periodicy = 1
    if arr.shape[0] > 1000:
        periodicy = arr.shape[0] // 1000
    i = 0
    si = 0
    short_arr = np.zeros(1000, dtype=np.float32)
    while i < arr.shape[0]-1:
        if i % periodicy == 0:
            short_arr = arr[i+1] - arr[i]
            si += 1
        i += 1
    return np.median(arr)

if __name__ == "__main__":
    
    print(f"\nTesting nearest_advocate:")
    size = 100
    np.random.seed(0)
    arr_ref = np.cumsum(np.random.random(size=size) + 0.5)
    arr_sig = arr_ref + np.random.normal(loc=0, scale=0.1, size=size) + np.pi 
    print(nearest_advocate(arr_ref, arr_sig, dist_max=None, dist_padding=None))
    
    
    print(f"\nTesting sparse_search_time_delta:")
    print(sparse_search_time_delta(arr_ref, arr_sig, td_min=-10, td_max=10, 
                                   sparse_factor=1, pps=10,
                                   dist_max=None, regulate_paddings=True, dist_padding=None)[:5])
    