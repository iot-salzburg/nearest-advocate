#!/usr/bin/env python
"""A time delay estimation method for event-based time-series in numba."""

from typing import Optional

import numpy as np

try:
    import numba
except ModuleNotFoundError:
    print("\nError: The 'numba' module is not installed.")
    print("You can install it using the following command:")
    print("pip install numba\n")


@numba.njit(parallel=False)
def nearest_advocate_single(arr_ref: 'np.ndarray[np.float32]', arr_sig: 'np.ndarray[np.float32]',
                            dist_max: float, dist_padding: float, regulate_paddings: bool = True
                            ) -> float:
    '''Calculates the synchronicity of two arrays of timestamps in terms of the mean of all minimal distances between each event in arr_sig and it's nearest advocate in arr_ref.
    arr_ref (np.array): Reference array or timestamps assumed to be correct
    arr_sig (np.array): Signal array of  timestamps, assumed to be shifted by an unknown constant time-delta
    dist_max (float): Maximal accepted distances, should be 1/4 of the median gap of arr_ref
    regulate_paddings (bool): Regulate non-overlapping events in arr_sig with a maximum distance of err_max, default True
    dist_padding (float): Distance assigned to non-overlapping (padding) events, should be 1/4 of the median gap of arr_ref. Only given if regulate_paddings is True
    '''
    # Assert input properties
    assert dist_max > 0.0          # maximal distance must be greater than 0.0
    if regulate_paddings:
        assert dist_padding > 0.0  # maximal distance for paddings must be greater than 0.0

    # store the lengths of the arrays
    l_arr_ref = len(arr_ref)
    l_arr_sig = len(arr_sig)

    ref_idx = 0              # index for arr_ref
    sig_idx = 0              # index for arr_sig
    counter = 0              # number of advocate events
    cum_distance = 0.0       # cumulative distances between advocate events

    # Step 1: cut leading reference timestamps without finding advocates
    while ref_idx+1 < l_arr_ref and arr_ref[ref_idx+1] <= arr_sig[sig_idx]:
        ref_idx += 1

    # return dist_max, if arr_ref ends before arr_sig starts
    if ref_idx+1 == l_arr_ref:
        return dist_max

    # Case: arr_ref[ref_idx] < arr_sig[sig_idx] < arr_ref[ref_idx+1]
    assert arr_ref[ref_idx+1] > arr_sig[sig_idx]

    # Step 2: count leading signal timestamps with finding advocates
    while sig_idx < l_arr_sig and arr_sig[sig_idx] < arr_ref[ref_idx]:
        # Invariant: arr_ref[ref_idx] < arr_sig[sig_idx] < arr_ref[ref_idx+1]
        if regulate_paddings:
            cum_distance += min(arr_ref[ref_idx]-arr_sig[sig_idx], dist_padding)
            counter += 1
        sig_idx += 1

    # return dist_max, if arr_sig ends before arr_ref starts
    if sig_idx == l_arr_sig:
        return dist_max

    # Step 3 (regular case) and step 4 (match trailing signal timestamps)
    while sig_idx < l_arr_sig:
        # Step 3: regular case
        if arr_sig[sig_idx] < arr_ref[-1]:
            # forward arr_ref and then arr_sig until regalar case
            while arr_ref[ref_idx+1] <= arr_sig[sig_idx] and ref_idx < l_arr_ref-1:
                ref_idx += 1
            if ref_idx < l_arr_ref:  # the first inequality broke
                # Invariant: arr_ref[ref_idx] <= arr_sig[sig_idx] < arr_ref[ref_idx+1]
                # assert arr_ref[ref_idx] <= arr_sig[sig_idx]
                # assert arr_sig[sig_idx] < arr_ref[ref_idx+1]
                cum_distance += min(arr_sig[sig_idx]-arr_ref[ref_idx], arr_ref[ref_idx+1]-arr_sig[sig_idx], dist_max)
                counter += 1
        # Step 4: match trailing reference timestamps with last signal timestamp
        elif regulate_paddings:
            # Invariant: arr_ref[ref_idx+1] <= arr_sig[sig_idx], given by the else case
            if arr_sig[sig_idx]-arr_ref[ref_idx+1] < dist_padding:
                cum_distance += arr_sig[sig_idx]-arr_ref[ref_idx+1]
                counter += 1
            else:
                # case with only dist_padding increments from now on
                cum_distance += (l_arr_sig - sig_idx) * dist_padding
                counter += (l_arr_sig - sig_idx)
                break  # stop, because the last values can be aggregated
        sig_idx += 1

    # return mean cumulative distance between found advocate events
    return cum_distance / counter


@numba.njit(parallel=False)
def nearest_advocate(arr_ref: np.ndarray, arr_sig: np.ndarray,
                     td_min: float, td_max: float, sps: Optional[float] = None,
                     sparse_factor: int = 1, dist_max: Optional[float] = None,
                     regulate_paddings: bool = True, dist_padding: Optional[float] = None
                     ) -> 'np.ndarray[(any, 2), np.float32]':
    '''Calculates the synchronicity of two arrays of timestamps for a search space between td_min and td_max in steps of 1/sps distance. The synchronicity is given by the mean of all minimal distances between each event in arr_sig and it's nearest advocate in arr_ref.

    Parameters
    ----------
    arr_ref : array_like
        Sorted reference array (1-D) with timestamps assumed to be correct.
    arr_sig : array_like
        Sorted signal array (1-D) of timestamps, assumed to be shifted by an unknown constant time-delta.
    td_min : float
        Lower bound of the search space for the time-shift.
    td_max : float
        Upper bound of the search space for the time-shift.
    sps : int, optional
        Number of investigated time-shifts per second. Default None: sets it at 10 divided by the median gap of each array.
    sparse_factor : int, optional
        Factor for the sparseness of `arr_sig` for the calculation, higher is faster at the cost of precision (default 1).
    dist_max : float
        Maximal accepted distances between two advocate events. Should be around 1/4 of the average gap of each array.
    regulate_paddings : bool, optional
        Regulate non-overlapping events in `arr_sig` with a maximum distance of dist_padding (default True).
    dist_padding : float, optional
        Distance assigned to non-overlapping (padding) events. Default None: 1/4 of the median gap of each array. Obsolete if `regulate_paddings` is False

    Returns
    -------
    time_shifts : array_like
        Two-columned 2-D array with evaluated time-shifts (from `td_min` to `td_max`) and their respective mean distances. The time-shift with the lowest mean distance is the estimation for the time delay between the arrays.

    References
    ----------
    C. Schranz, S. Mayr, "Ein neuer Algorithmus zur Zeitsynchronisierung von Ereignis-basierten Zeitreihendaten als Alternative zur Kreuzkorrelation", Spinfortec (Chemnitz 2022). :doi:`10.5281/zenodo.7370958`

    Examples
    --------
    >>> import numpy as np
    >>> import nearest_advocate
    >>> N = 10_000

    Create a reference array whose events differences are sampled from a normal distribution. The signal array is the reference but shifted by `np.pi` and addional gaussian noise. The event-timestamps of both arrays must be sorted.

    >>> arr_ref = np.sort(np.cumsum(np.random.normal(loc=1, scale=0.25, size=N)))
    >>> arr_sig = np.sort(arr_ref + np.pi + np.random.normal(loc=0, scale=0.1, size=N))

    The function `nearest_advocate.nearest_advocate` returns a two-columned array with all investigated time-shifts and their mean distances, i.e., the measure of the synchronicity between both array (lower is better).

    >>> time_shifts = nearest_advocate.nearest_advocate(arr_ref=arr_ref, arr_sig=arr_sig, td_min=-60, td_max=60, sps=10)
    >>> time_shift, min_mean_dist = time_shifts[np.argmin(time_shifts[:,1])]
    >>> print(f"Found an optimum at {time_shift:.4f}s with a minimal mean distance of {min_mean_dist:.6f}s")
    Found an optimum at 3.15s with a minimal mean distance of 0.079508s

    Plot the resulting table

    >>> import matplotlib.pyplot as plt
    >>> plt.plot(time_shifts[:,0], time_shifts[:,1], color="steelblue", label="Mean distance")
    >>> plt.vlines(x=time_shift, ymin=min_mean_dist, ymax=np.mean(time_shifts[:,1]), color="firebrick", label=f"Shift = {time_shift:.2f}s")
    >>> plt.xlim(time_shift-4, time_shift+4)
    >>> plt.xlabel("Time delay (s)")
    >>> plt.ylabel("Mean distance (s)")
    >>> plt.legend(loc="lower right")
    >>> plt.show()
    '''
    # set default values if unset
    if sps is None or sps <= 0.0:
        sps_float = 10 / min(np.median(np.diff(arr_ref)), np.median(np.diff(arr_sig)))
    else:
        sps_float = sps
    if dist_max is None or dist_max <= 0.0:
        dist_max_float = min(np.median(np.diff(arr_ref)), np.median(np.diff(arr_sig)))/4
    else:
        dist_max_float = dist_max
    if dist_padding is None or dist_padding <= 0.0:
        dist_padding_float = min(np.median(np.diff(arr_ref)), np.median(np.diff(arr_sig)))/4
    else:
        dist_padding_float = dist_padding

    # Create a copy of arr_sig and make sparse if set
    if int(sparse_factor) > 1:
        probe = arr_sig.copy()[int(sparse_factor/2)::int(sparse_factor)]
    else:
        probe = arr_sig.copy()

    # Create an T x 2 matrix to store the evaluated time-shifts with their respective mean distance
    np_nearest = np.empty((int((td_max-td_min)*sps_float), 2), dtype=np.float32)
    np_nearest[:, 0] = np.arange(td_min, td_max-1/sps_float+1e-12, 1/sps_float)

    # Calculate the mean distance for all time-shifts in the search space.
    idx = 0
    while idx < np_nearest.shape[0]:
        # calculate the nearest advocate criteria
        np_nearest[idx, 1] = nearest_advocate_single(
             arr_ref,
             probe-np_nearest[idx, 0],  # the signal array is shifted by a time-delta
             dist_max=dist_max_float,
             regulate_paddings=regulate_paddings, dist_padding=dist_padding_float)
        idx += 1
    # The shift with the lowest mean distance is the best fit for the time-shift
    return np_nearest


if __name__ == "__main__":
    print("\nTesting Nearest Advocate for a single shift:")
    SIZE = 100
    np.random.seed(0)
    arr_reference = np.cumsum(np.random.random(size=SIZE) + 0.5)
    arr_signal = arr_reference + np.random.normal(loc=0, scale=0.1, size=SIZE) + np.pi
    print(nearest_advocate_single(arr_reference, arr_signal, dist_max=0.25, dist_padding=0.25))

    print("\nTesting Nearest Advocate for a search space:")
    print(nearest_advocate(arr_reference, arr_signal, td_min=-10, td_max=10,
                           sparse_factor=1, sps=None,
                           dist_max=None, regulate_paddings=True, dist_padding=None)[:5])
