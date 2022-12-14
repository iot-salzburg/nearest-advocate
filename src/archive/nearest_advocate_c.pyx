import numpy as np
cimport numpy as np


def _nearest_advocate_single(arr_ref: np.ndarray, arr_sig: np.ndarray, 
                              dist_max: float, regulate_paddings: bool, dist_padding: float):
    """Post-hoc synchronization method for event-based time-series data.
    
    Calculates the synchronicity of two arrays of timestamps in terms of the mean of all minimal distances between each event in arr_sig and its nearest advocate in arr_ref.

    Parameters
    ----------
    arr_ref : array_like
        Reference array (1-D) or timestamps assumed to be correct.
    arr_sig : array_like
        Signal array (1-D) of timestamps, assumed to be shifted by an unknown constant time-delta.
    dist_max : float
        Maximal accepted distances between two advocate events. It should be around 1/4 of the median gap of `arr_ref`.
    regulate_paddings : bool
        Regulate non-overlapping events in `arr_sig` with a maximum distance of dist_padding, default: True.
    dist_padding : float
        Distance assigned to non-overlapping (padding) events. It should be around 1/4 of the median gap of `arr_ref`. Obsolete if `regulate_paddings` is False

    Returns
    -------
    mean_dist : float
        Mean distance between all pairs of advocates, quantity of the synchronicity between the two arrays.

    See Also
    --------
    nearest_advocate

    Notes
    -----
    .. versionadded:: 1.9.4

    References
    ----------
    C. Schranz, S. Mayr, "Ein neuer Algorithmus zur Zeitsynchronisierung von Ereignis-basierten Zeitreihendaten als Alternative zur Kreuzkorrelation", Spinfortec (Chemnitz 2022). :doi:`10.5281/zenodo.7370958`

    Examples
    --------
    >>> import numpy as np
    >>> np.random.seed(0)
    >>> from scipy.signal import _nearest_advocate_single
    >>> N = 100_000 
    >>> DEF_DIST = 0.25    
    
    Create a reference array whose events differences are sampled from a normal distribution. The signal array is the reference but shifted by `np.pi` and addional gaussian noise. The event-timestamps of both arrays must be sorted.
    
    >>> arr_ref = np.sort(np.cumsum(np.random.normal(loc=1, scale=0.25, size=N))).astype(np.float32)
    >>> arr_sig = np.sort(arr_ref + np.pi + np.random.normal(loc=0, scale=0.1, size=N)).astype(np.float32)

    The function `_nearest_advocate_single` returns a measure of the synchronicity between both array (lower is better). 

    >>> _nearest_advocate_single(arr_ref=arr_ref, arr_sig=arr_sig, 
                                  dist_max=DEF_DIST, regulate_paddings=True, 
                                  dist_padding=DEF_DIST)
    0.18436528742313385
    """
    # convert the event-based arrays to cython 1-D ndarray
    cdef np.ndarray[np.float32_t, ndim=1, cast=True] arr_ref_c = arr_ref
    cdef np.ndarray[np.float32_t, ndim=1, cast=True] arr_sig_c = arr_sig
    
    assert dist_max > 0.0  # Maximal accepted distances between two advocate events. (see docs)
    if regulate_paddings:
        assert dist_padding > 0.0  # Distance assigned to non-overlapping events. (see docs)
    
    # calculate and return the mean distance between both arrays
    return _nearest_advocate_single_c(arr_ref_c, arr_sig_c, 
                                      dist_max=dist_max, regulate_paddings=regulate_paddings, 
                                      dist_padding=dist_padding)

cdef np.float32_t _nearest_advocate_single_c(np.ndarray[np.float32_t, ndim=1] arr_ref_c, 
                                             np.ndarray[np.float32_t, ndim=1] arr_sig_c, 
                                             np.float32_t dist_max=0.0, bint regulate_paddings=True, 
                                             np.float32_t dist_padding=0.0):

    # store the lengths of the arrays
    cdef np.uint32_t l_arr_ref_c = arr_ref_c.shape[0]
    cdef np.uint32_t l_arr_sig_c = arr_sig_c.shape[0]
    
    cdef np.uint32_t ref_idx = 0         # index for arr_ref_c
    cdef np.uint32_t sig_idx = 0         # index for arr_sig_c
    cdef np.uint32_t counter = 0         # number of advocate events
    cdef np.float32_t cum_distance = 0.0 # cumulative distances between advocate events

    # Step 1: cut leading reference timestamps without finding advocates
    while ref_idx+1 < l_arr_ref_c and arr_ref_c[ref_idx+1] <= arr_sig_c[sig_idx]:
        ref_idx += 1
        
    # return dist_max, if arr_ref_c ends before arr_sig_c starts
    if ref_idx+1 == l_arr_ref_c:
        return dist_max
    
    # Case: arr_ref_c[ref_idx] < arr_sig_c[sig_idx] < arr_ref_c[ref_idx+1]
    assert arr_ref_c[ref_idx+1] > arr_sig_c[sig_idx]
    
    # Step 2: count leading signal timestamps with finding advocates
    while sig_idx < l_arr_sig_c and arr_sig_c[sig_idx] < arr_ref_c[ref_idx]:
        # Invariant: arr_ref_c[ref_idx] < arr_sig_c[sig_idx] < arr_ref_c[ref_idx+1]
        if regulate_paddings:
            cum_distance += min(arr_ref_c[ref_idx]-arr_sig_c[sig_idx], dist_padding)
            counter += 1
        sig_idx += 1
    
    # return dist_max, if arr_sig_c ends before arr_ref_c starts
    if sig_idx == l_arr_sig_c:
        return dist_max        
    
    # Step 3 (regular case) and step 4 (match trailing signal timestamps)
    while sig_idx < l_arr_sig_c:
        # Step 3: regular case
        if arr_sig_c[sig_idx] < arr_ref_c[-1]:
            # forward arr_ref_c and then arr_sig_c until regalar case
            while ref_idx+1 < l_arr_ref_c and arr_ref_c[ref_idx+1] <= arr_sig_c[sig_idx]:
                ref_idx += 1
            if ref_idx+1 >= l_arr_ref_c: 
                sig_idx += 1
                continue
            # Invariant: arr_ref_c[ref_idx] < arr_sig_c[sig_idx] < arr_ref_c[ref_idx+1]
            # assert arr_ref_c[ref_idx] <= arr_sig_c[sig_idx]
            # assert arr_sig_c[sig_idx] < arr_ref_c[ref_idx+1]
            
            cum_distance += min(arr_sig_c[sig_idx]-arr_ref_c[ref_idx], arr_ref_c[ref_idx+1]-arr_sig_c[sig_idx], dist_max) 
            counter += 1
        # Step 4: match trailing reference timestamps with last signal timestamp
        elif regulate_paddings:  
            # Invariant: arr_ref_c[ref_idx+1] <= arr_sig_c[sig_idx], given by the else case
            if arr_sig_c[sig_idx]-arr_ref_c[ref_idx+1] < dist_padding:
                cum_distance += arr_sig_c[sig_idx]-arr_ref_c[ref_idx+1]
                counter += 1
            else: 
                # case with only dist_padding increments from now on
                cum_distance += (l_arr_sig_c - sig_idx) * dist_padding
                counter += (l_arr_sig_c - sig_idx)
                break # stop, because the last values can be aggregated
                
        sig_idx += 1

    # return mean cumulative distance between found advocate events
    return cum_distance / counter

def nearest_advocate(arr_ref: np.ndarray, arr_sig: np.ndarray, 
                       td_min: float, td_max: float, sps: float=10, sparse_factor: int=1,
                       dist_max: float=0.0, regulate_paddings: bool=True, dist_padding: float=0.0):
    """Post-hoc synchronization method for event-based time-series data.
    
    Calculates the synchronicity of two arrays of timestamps for a search space between td_min and td_max with a precision of 1/sps. The synchronicity is given by the mean of all minimal distances between each event in arr_sig and its nearest advocate in arr_ref.

    Parameters
    ----------
    arr_ref : array_like
        Reference array (1-D) or timestamps assumed to be correct.
    arr_sig : array_like
        Signal array (1-D) of timestamps, assumed to be shifted by an unknown constant time-delta.    
    td_min : float
        Lower bound of the search space for the time-shift.
    td_max : float 
        Upper bound of the search space for the time-shift.
    sps : int, optional
        Number of investigated time-shifts per second, should be higher than 10 times the number of median gap of `arr_ref` (default 10).
    sparse_factor : int, optional
        Factor for the sparseness of `arr_sig` for the calculation, higher is faster but may be less accurate (default 1).
    dist_max : float, optional
        Maximal accepted distances between two advocate events. It should be around 1/4 of the median gap of `arr_ref` (default).
    regulate_paddings : bool, optional
        Regulate non-overlapping events in `arr_sig` with a maximum distance of dist_padding (default True).
    dist_padding : float, optional
        Distance assigned to non-overlapping (padding) events. It should be around 1/4 of the median gap of `arr_ref` (default). Obsolete if `regulate_paddings` is False

    Returns
    -------
    time_shifts : array_like
        Two-columned 2-D array with evaluated time-shifts (between `td_min` and `td_max`) and the respective mean distances. The time-delta with the lowest mean distance is the estimation for the time-shift between the two arrays.

    Notes
    -----
    .. versionadded:: 1.9.4

    References
    ----------
    C. Schranz, S. Mayr, "Ein neuer Algorithmus zur Zeitsynchronisierung von Ereignis-basierten Zeitreihendaten als Alternative zur Kreuzkorrelation", Spinfortec (Chemnitz 2022). :doi:`10.5281/zenodo.7370958`

    Examples
    --------
    >>> import numpy as np 
    >>> np.random.seed(0)
    >>> from scipy.signal import nearest_advocate
    >>> N = 100_000 
    >>> DEF_DIST = 0.25    
    
    Create a reference array whose events differences are sampled from a normal distribution. The signal array is the reference but shifted by `np.pi` and addional gaussian noise. The event-timestamps of both arrays must be sorted.
    
    >>> arr_ref = np.sort(np.cumsum(np.random.normal(loc=1, scale=0.25, size=N))).astype(np.float32)
    >>> arr_sig = np.sort(arr_ref + np.pi + np.random.normal(loc=0, scale=0.1, size=N)).astype(np.float32)

    The function `nearest_advocate` returns a two-columned array with all investigated time-shifts and their mean distances, i.e., the measure of the synchronicity between both array (lower is better). 
    
    >>> time_shifts = nearest_advocate(arr_ref=arr_ref, arr_sig=arr_sig, 
                                       td_min=-60, td_max=60, sps=20, sparse_factor=1, 
                                       dist_max=DEF_DIST, regulate_paddings=True,
                                       dist_padding=DEF_DIST)
    >>> time_shift, min_mean_dist = time_shifts[np.argmin(time_shifts[:,1])]
    >>> print(time_shift, min_mean_dist)
    3.15, 0.07941238
    """
    # convert the event-based arrays to cython 1-D ndarray
    cdef np.ndarray[np.float32_t, ndim=1, cast=True] arr_ref_c = arr_ref
    cdef np.ndarray[np.float32_t, ndim=1, cast=True] arr_sig_c = arr_sig
    
    # create a kx2 array to store the time-shifts with their respective synchronicity
    cdef np.ndarray[np.float32_t, ndim=2] np_nearest_c = np.empty((int((td_max-td_min)*sps), 2), 
                                                                  dtype=np.float32)
    # fill the first column with the time-shifts to test
    idx = 0
    td = td_min
    while idx < np_nearest_c.shape[0]:
        np_nearest_c[idx, 0] = td
        td += 1/sps
        idx += 1
    
    # call and return the cython function
    return _nearest_advocate_c(arr_ref_c, arr_sig_c, np_nearest_c=np_nearest_c, 
                               td_min=td_min, td_max=td_max, sps=sps, sparse_factor=sparse_factor, 
                               dist_max=dist_max, regulate_paddings=regulate_paddings, 
                               dist_padding=dist_padding)

cdef np.ndarray[np.float32_t, ndim=2] _nearest_advocate_c(
    np.ndarray[np.float32_t, ndim=1] arr_ref_c, np.ndarray[np.float32_t, ndim=1] arr_sig_c, 
    np.ndarray[np.float32_t, ndim=2] np_nearest_c, np.float32_t td_min, np.float32_t td_max, 
    np.float32_t sps=10.0, np.int32_t sparse_factor=1, 
    np.float32_t dist_max=0.0, bint regulate_paddings=1, np.float32_t dist_padding=0.0):
    # set the default values for dist_max, dist_padding relative if not set
    if dist_max == 0.0:
        dist_max = np.median(np.diff(arr_ref_c))/4
    if dist_padding == 0.0:
        dist_padding = np.median(np.diff(arr_ref_c))/4
        
    # Random subsample
    if sparse_factor > 1:
        arr_sig_c = arr_sig_c[sparse_factor//2::sparse_factor]

    # Calculate the mean distance for all time-shifts in the search space. 
    # The shift with the lowest mean distance is the best fit for the time-shift
    idx = 0
    while idx < np_nearest_c.shape[0]:
        # calculate the nearest advocate criteria
        np_nearest_c[idx,1] = _nearest_advocate_single_c(
             arr_ref_c, 
             arr_sig_c-np_nearest_c[idx,0],  # the signal array is shifted by a time-delta
             dist_max=dist_max, regulate_paddings=regulate_paddings, 
             dist_padding=dist_padding)
        idx += 1
    return np_nearest_c
