def nearest_advocate_inner(
    arr_ref: np.ndarray, arr_sig: np.ndarray, dist_max: float):
    # arr_sig: signal array is shifted by a time offset
    ref_idx = 0              # index for arr_ref
    sig_idx = 0              # index for arr_sig
    counter = 0              # number of advocate events
    cum_distance = 0.0       # cumulative distance
    # Skip indices until matching
    ref_idx, sig_idx = forward_until_overlapping(arr_ref, arr_sig)
    # Regular case for nearest advocate match
    while sig_idx < len(arr_sig):
        if arr_sig[sig_idx] < arr_ref[-1]:
            while arr_ref[ref_idx+1] <= arr_sig[sig_idx] and ref_idx < l_arr_ref-1:
                ref_idx += 1
            if ref_idx < l_arr_ref:  # the first inequality broke
                # Invariant: arr_ref[ref_idx] <= arr_sig[sig_idx]
                #       and: arr_sig[sig_idx] < arr_ref[ref_idx+1]
                cum_distance += min(arr_sig[sig_idx]-arr_ref[ref_idx], 
                        arr_ref[ref_idx+1]-arr_sig[sig_idx], dist_max) 
                counter += 1      # increment number of matches
        sig_idx += 1  # increment signal index        
    # return mean cumulative distance of advocate events
    return cum_distance / counter


def nearest_advocate(arr_ref: np.ndarray, arr_sig: np.ndarray, 
                     td_min: float, td_max: float, sps: float=10, sparse_factor: int=1, 
                     dist_max: float=0.0, regulate_paddings: bool=True, dist_padding: float=0.0):
    '''Calculates the synchronicity of two arrays of timestamps for a search space between td_min and td_max with a precision of 1/sps. The synchronicity is given by the mean of all minimal distances between each event in arr_sig and it's nearest advocate in arr_ref.
    arr_ref (np.array): Reference array or timestamps assumed to be correct
    arr_sig (np.array): Signal array of  timestamps, assumed to be shifted by an unknown constant time-delta
    td_min (float): lower bound of the search space for the time-shift
    td_max (float): upper bound of the search space for the time-shift
    sps (int): number of investigated time-shifts per second, should be higher than 10 times the number of median gap of arr_ref (default 10).
    sparse_factor (int): factor for the sparseness of arr_sig for the calculation, higher is faster but may be less accurate (default 1)
    dist_max (None, float): Maximal accepted distances, default None: 1/4 of the median gap of arr_ref
    dist_padding (None, float): Assumed distances of non-overlapping (padding) matches, default None: 1/4 of the median gap of arr_ref
    regulate_paddings (bool): regulate non-overlapping events in arr_sig with a maximum distance of err_max
    '''
    # set the default values for dist_max, dist_padding relative if not set
    # TODO improve default value: min(np.median(np.diff(arr_sig)), np.median(np.diff(arr_ref))) / 4
    if dist_max == 0.0:
        dist_max = min(np.median(np.diff(arr_ref))/4, np.median(np.diff(arr_sig))/4)
    if dist_padding == 0.0:
        dist_padding = min(np.median(np.diff(arr_ref))/4, np.median(np.diff(arr_sig))/4)
        
    # Random subsample and create a copy of arr_sig once, as it could lead to problems otherwise
    if sparse_factor > 1:
        probe = arr_sig.copy()[sparse_factor//2::sparse_factor]
    else:
        probe = arr_sig.copy()
    
    # Create an k x 2 matrix to store the investigated time-shifts and their respective mean distance
    np_nearest = np.empty((int((td_max-td_min)*sps), 2), dtype=np.float32)
    np_nearest[:, 0] = np.arange(td_min, td_max, 1/sps)
    
    # Calculate the mean distance for all time-shifts in the search space. 
    # The shift with the lowest mean distance is the best fit for the time-shift
    idx = 0
    while idx < np_nearest.shape[0]:
        # calculate the nearest advocate criteria
        np_nearest[idx,1] = nearest_advocate_single(
             arr_ref, 
             probe-np_nearest[idx,0],  # the signal array is shifted by a time-delta
             dist_max=dist_max, regulate_paddings=regulate_paddings, dist_padding=dist_padding)
        idx += 1
    return np_nearest


if __name__ == "__main__":
    print(f"\nTesting nearest_advocate:")
    size = 100
    np.random.seed(0)
    arr_ref = np.cumsum(np.random.random(size=size) + 0.5)
    arr_sig = arr_ref + np.random.normal(loc=0, scale=0.1, size=size) + np.pi 
    print(nearest_advocate_single(arr_ref, arr_sig, dist_max=0.25, dist_padding=0.25))
    
    print(f"\nTesting sparse_search_time_delta:")
    print(nearest_advocate(arr_ref, arr_sig, td_min=-10, td_max=10, 
                           sparse_factor=1, sps=10,
                           dist_max=0.0, regulate_paddings=True, dist_padding=0.0)[:5])
    