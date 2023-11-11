#!/usr/bin/env python
"""A time delay estimation method for event-based time-series in Python."""


import numpy as np


def nearest_advocate_single(arr_ref: np.ndarray, arr_sig: np.ndarray, dist_max: float):
    '''Calculates the synchronicity of two arrays of timestamps in terms of the mean of all minimal distances between each event in arr_sig and it's nearest advocate in arr_ref.
    arr_ref (np.array): Reference array or timestamps assumed to be correct
    arr_sig (np.array): Signal array of  timestamps, assumed to be shifted by an unknown constant time-delta
    dist_max (float): Maximal accepted distances, should be 1/4 of the median gap of arr_ref
    '''
    # Assert input properties
    assert dist_max > 0.0          # maximal distance must be greater than 0.0

    # store the lengths of the arrays
    l_arr_ref = len(arr_ref)
    l_arr_sig = len(arr_sig)
    assert l_arr_ref > 2 and l_arr_sig > 2

    ref_idx = 0              # index for arr_ref
    sig_idx = 0              # index for arr_sig
    cum_distance = 0.0       # cumulative distances between advocate events

    # Case 1: cut leading reference timestamps without finding advocates
    while ref_idx + 1 < l_arr_ref and arr_ref[ref_idx+1] <= arr_sig[sig_idx]:
        ref_idx += 1


    # Case 2 and Case 3: At least one signal ts is not matched with the final reference ts
    if ref_idx + 1 < l_arr_ref:
        # assert arr_sig[sig_idx] < arr_ref[ref_idx+1]

        # Case 2: match leading signal timestamps with nearest advocates
        while sig_idx < l_arr_sig and arr_sig[sig_idx] < arr_ref[ref_idx]:
            # Loop invariant: arr_sig[sig_idx] < arr_ref[ref_idx]
            cum_distance += min(arr_ref[ref_idx]-arr_sig[sig_idx], dist_max)
            sig_idx += 1
        # if arr_sig ends before arr_ref starts, other cases are skipped as sig_idx = l_arr_sig

        # Case 3: Regular case of
        while sig_idx < l_arr_sig and ref_idx < l_arr_ref - 1:
            # forward arr_ref and then arr_sig until the invariant holds
            while ref_idx < l_arr_ref - 1 and arr_ref[ref_idx+1] <= arr_sig[sig_idx]:
                ref_idx += 1
            if ref_idx < l_arr_ref - 1:
                # Invariant: arr_ref[ref_idx] <= arr_sig[sig_idx] < arr_ref[ref_idx+1]
                # assert arr_ref[ref_idx] <= arr_sig[sig_idx] + 1e-12
                # assert arr_sig[sig_idx] < arr_ref[ref_idx+1] + 1e-12
                cum_distance += min(arr_sig[sig_idx]-arr_ref[ref_idx], arr_ref[ref_idx+1]-arr_sig[sig_idx], dist_max)
                sig_idx += 1

    # Case 4: match trailing signal timestamps with last reference ts
    while sig_idx < l_arr_sig:
        # Invariant: arr_ref[-1] <= arr_sig[sig_idx]
        # assert arr_ref[-1] <= arr_sig[sig_idx] + 1e-12
        cum_distance += min(arr_sig[sig_idx]-arr_ref[-1], dist_max)
        sig_idx += 1

    # return mean cumulative distance between found advocate events
    assert cum_distance / l_arr_sig <= dist_max + 1e-12
    return cum_distance / l_arr_sig


def nearest_advocate(arr_ref: np.ndarray, arr_sig: np.ndarray, td_min: float, td_max: float,
                     dist_max: float=0.0, sps: float=10, sparse_factor: int=1,
                     symmetric: bool=False):
    '''Calculates the synchronicity of two arrays of timestamps for a search space between td_min and td_max in steps of 1/sps distance. The synchronicity is given by the mean of all minimal distances between each event in arr_sig and it's nearest advocate in arr_ref.

    Parameters
    ----------
    arr_ref : array_like
        Sorted reference array (1-D) with timestamps assumed to be correct.
    arr_sig : array_like
        Sorted signal array (1-D) of timestamps, assumed to be shifted by an unknown constant time-delta.
    dist_max : float
        Maximal accepted distances between two advocate events. Should be around 1/4 of the average gap of each array.
    td_min : float
        Lower bound of the search space for the time-shift.
    td_max : float
        Upper bound of the search space for the time-shift.
    sps : int, optional
        Number of investigated time-shifts per second. Default None: sets it at 10 divided by the median gap of each array.
    sparse_factor : int, optional
        Factor for the sparseness of `arr_sig` for the calculation, higher is faster at the cost of precision (default 1).
    symmetric : bool
        Perform the Nearest Advocate algorithm symmetrically, i.e., both orders of the arrays and the results are averaged (default False).

    Returns
    -------
    time_shifts : array_like
        Two-columned 2-D array with evaluated time-shifts (from `td_min` to `td_max`) and their respective mean distances. The time-shift with the lowest mean distance is the estimation for the time delay between the arrays.

    '''
    # If symmetric Nearest Advocate, call the algorithm recursively
    if symmetric:
        left = nearest_advocate(
            arr_ref, arr_sig, dist_max=dist_max, td_min=td_min, td_max=td_max,
             sps=sps, sparse_factor=sparse_factor, symmetric=False
             )
        right = nearest_advocate(
            arr_sig, arr_ref, dist_max=dist_max, td_min=td_min, td_max=td_max,
             sps=sps, sparse_factor=sparse_factor, symmetric=False
             )
        return (left + right) / 2

    # Assert properties of the arrays
    assert len(arr_ref) >= 2 and len(arr_sig) >= 2
    assert np.all(np.diff(arr_ref)>0), "The reference array must be strictly increasing (without duplicates)."
    assert np.all(np.diff(arr_sig)>0), "The signal array must be strictly increasing (without duplicates)."

    # set default values if unset
    if sps is None or sps <= 0.0:
        sps_float = 10 / min(np.median(np.diff(arr_ref)), np.median(np.diff(arr_sig)))
    else:
        sps_float = sps
    if dist_max is None:
        dist_max_float = min(np.median(np.diff(arr_ref)), np.median(np.diff(arr_sig))) / 4
    else:
        assert dist_max > 0.0
        dist_max_float = dist_max

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
             dist_max=dist_max_float)
        idx += 1
    # The shift with the lowest mean distance is the best fit for the time-shift
    return np_nearest


if __name__ == "__main__":
    print("\nTesting nearest_advocate:")
    SIZE = 100
    np.random.seed(0)
    arr_reference = np.cumsum(np.random.random(size=SIZE) + 0.5)
    arr_signal = arr_reference + np.random.normal(loc=0, scale=0.1, size=SIZE) + np.pi
    print(nearest_advocate_single(arr_reference, arr_signal, dist_max=0.25))

    print("\nTesting Nearest Advocate for a search space:")
    print(nearest_advocate(arr_reference, arr_signal, dist_max=None,
                           td_min=-10, td_max=10, sparse_factor=1, sps=None, symmetric=True)[:5])
