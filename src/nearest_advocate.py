#!/usr/bin/env python
"""A time delay estimation method for event-based time-series."""

__author__ = "Christoph Schranz"
__copyright__ = "Copyright 2025, Salzburg Research"
__version__ = "0.2.0.5"
__maintainer__ = "Christoph Schranz, Mathias Schmoigl-Tonis"
__credits__ = ["Christoph Schranz", "Mathias Schmoigl-Tonis"]


from typing import Optional

import numpy as np

# Using the numba-implementation right now
from nearest_advocate_numba import nearest_advocate as _nearest_advocate
from nearest_advocate_numba import nearest_advocate_single as _nearest_advocate_single


def nearest_advocate_single(arr_ref: 'np.ndarray[np.float32]', arr_sig: 'np.ndarray[np.float32]',
                            dist_max: float, symmetric: bool = False) -> float:
    '''Calculates the synchronicity between two arrays of timestamps in terms of the mean of all minimal distances between each event in arr_sig and it's nearest advocate event in arr_ref.

    Parameters
    ----------
    arr_ref : array_like
        Array (1-D) of sorted timestamps that are considered as reference times.
    arr_sig : array_like
        Array (1-D) of sorted timestamps that are assumed to be shifted (i.e. delayed) by an unknown but constant time-delta.
    dist_max : float
        Important hyperparameter which caps the maximal distance between any signal timestamp and its nearest reference timestamp. Too low values decrease robustness for noisy timestamps, too high values over-smooth the estimation's minima. Default None: 1/4 of the smaller median inter-event interval of the arrays (1/4 * min(np.median(np.diff(arr_ref)), np.median(np.diff(arr_sig)))).
    symmetric : bool
        Perform the Nearest Advocate algorithm symmetrically, i.e., both orders of the arrays and the results are averaged. This is better but assumes no missing events in `arr_sig` (default False).
        
    Returns
    -------
    mean_distance : float
        The mean distance of each event in `arr_sig` to its nearest advocate event in `arr_ref`.
    '''
    # cast the arrays to numpy arrays
    arr_ref_ = np.array(arr_ref)
    arr_sig_ = np.array(arr_sig)
    
    # cast the arrays to np.ndarray, subtract the minimal timestamp to avoid floating point error
    min_event_time = min(arr_ref_[0], arr_sig_[0])
    arr_ref_ = np.array(arr_ref_ - min_event_time, dtype=np.float32)
    arr_sig_ = np.array(arr_sig_ - min_event_time, dtype=np.float32)

    # If symmetric Nearest Advocate, call the algorithm recursively
    if symmetric:
        left = _nearest_advocate_single(arr_ref_, arr_sig_, dist_max=dist_max)
        right = _nearest_advocate_single(arr_sig_, arr_ref_, dist_max=dist_max)
        return left + right
    
    return _nearest_advocate_single(arr_ref=arr_ref_, arr_sig=arr_sig_, dist_max=dist_max)


def nearest_advocate(arr_ref: 'np.ndarray[np.float32]', arr_sig: 'np.ndarray[np.float32]',
                     td_min: float, td_max: float, dist_max: Optional[float] = None,
                     sps: Optional[float] = None, sparse_factor: int = 1, symmetric=False
                     ) -> 'np.ndarray[(any, 2), np.float32]':
    '''Calculates the synchronicity between two arrays of timestamps for a search space that ranges from td_min to td_max with a stepwidth of 1/sps. The synchronicity measures is given by the mean of all minimal distances between each event in arr_sig and it's nearest advocate event in arr_ref.

    Parameters
    ----------
    arr_ref : array_like
        Array (1-D) of sorted timestamps that are considered as reference times.
    arr_sig : array_like
        Array (1-D) of sorted timestamps that are assumed to be shifted (i.e. delayed) by an unknown but constant time-delta.
    td_min : float
        Lower bound of the one-dimensional search space of time-deltas.
    td_max : float
        Upper bound of the one-dimensional search space of time-deltas.
    dist_max : float
        Important hyperparameter which caps the maximal distance between any signal timestamp and its nearest reference timestamp. Too low values decrease robustness for noisy timestamps, too high values over-smooth the estimation's minima. Default None: 1/4 of the smaller median inter-event interval of the arrays (1/4 * min(np.median(np.diff(arr_ref)), np.median(np.diff(arr_sig)))).
    sps : float, optional
        Resolution or number of investigated time-shifts per second. Higher values increases the estimation's precision by the cost of performance. Default None: sets it at 100 divided by the median gap of each array.
    sparse_factor : int, optional
        Factor to process a sparse `arr_sig` by taking every k-th (k>0) value into account; higher is faster at the (not recommendable) cost of precision (default 1).
    symmetric : bool
        Perform the Nearest Advocate algorithm symmetrically, i.e., both orders of the arrays and the results are averaged. This is better but assumes no missing events in `arr_sig` (default False).

    Returns
    -------
    time_shifts : array_like
        Two-columned 2-D array: In the first column all evaluated time-shifts ranging from `td_min` to `td_max` with a step-width of sps. In the second column the respective mean distances of each event in `arr_sig` to its nearest advocate event in `arr_ref` for the given time-delta. The time-shift with the lowest mean distance is the optimal estimation for the time-delay of the signal array relative to the reference array.

    References
    ----------
    Schranz, C., Mayr, S., Bernhart, S. et al. Nearest advocate: a novel event-based time delay estimation algorithm for multi-sensor time-series data synchronization. EURASIP J. Adv. Signal Process. 2024, 46 (2024). :doi:`10.1186/s13634-024-01143-1`

    Examples
    --------
    >>> import numpy as np
    >>> import nearest_advocate

    Create a reference array of timestamps whose inter-events intervals are sampled from a normal distribution with `mu=1` and `std=0.25`. The signal array is the reference but shifted by `np.pi` and addional gaussian noise with `std=0.1`. The event-timestamps of both arrays must be sorted.

    >>> arr_ref = np.sort(np.cumsum(np.random.normal(loc=1, scale=0.25, size=1000)))
    >>> arr_sig = np.sort(arr_ref + np.pi + np.random.normal(loc=0, scale=0.1, size=1000))
    >>> print(arr_sig[:8])
    [ 4.63820201  5.77189243  6.88509806  8.49802424  9.97724767 10.73027086
 12.00172233 12.7279979 ]

    The function `nearest_advocate.nearest_advocate` returns a two-columned array with all investigated time-shifts (i.e. time-delays) and their respective `mean distance (over all nearest advocates)`, i.e., this algorithm's measure of the synchronicity between both array. A lower mean distance is better.

    >>> time_shifts = nearest_advocate.nearest_advocate(arr_ref=arr_ref, arr_sig=arr_sig, dist_max=0.25, td_min=-60, td_max=60, sps=100)
    >>> time_shift, min_mean_dist = time_shifts[np.argmin(time_shifts[:,1])]
    >>> print(f"Found an optimum at {time_shift:.4f} s with a minimal mean distance of {min_mean_dist:.6f} s")
    Found an optimum at 3.1400 s with a minimal mean distance of 0.076846 s.

    Plot the resulting table

    >>> import matplotlib.pyplot as plt
    >>> plt.plot(time_shifts[:,0], time_shifts[:,1], color="steelblue", label="Mean distance")
    >>> plt.vlines(x=time_shift, ymin=min_mean_dist, ymax=np.mean(time_shifts[:,1]), color="firebrick", label=f"Shift = {time_shift:.2f} s")
    >>> plt.xlim(time_shift-4, time_shift+4)
    >>> plt.xlabel("Time delay (s)")
    >>> plt.ylabel("Mean distance (s)")
    >>> plt.legend(loc="lower right")
    >>> plt.show()
    '''
    # cast the arrays to numpy arrays
    arr_ref_ = np.array(arr_ref)
    arr_sig_ = np.array(arr_sig)

    # Assert input properties
    assert len(arr_ref_.shape) == 1  # must be a 1D array
    assert len(arr_sig_.shape) == 1  # must be a 1D array
    assert arr_ref_.shape[0] > 1    # reference array must have >= 2 elements
    assert arr_sig_.shape[0] > 1    # signal array must have >= 2 elements
    
    # cast the arrays to np.ndarray, subtract the minimal timestamp to avoid floating point error
    min_event_time = min(arr_ref_[0], arr_sig_[0])
    arr_ref_ = np.array(arr_ref_ - min_event_time, dtype=np.float32)
    arr_sig_ = np.array(arr_sig_ - min_event_time, dtype=np.float32)
    
    assert isinstance(td_min, (int, float))
    assert isinstance(td_max, (int, float))
    assert isinstance(sparse_factor, int) and sparse_factor > 0

    # set default values if unset
    if sps is None or sps <= 0.0:
        sps_float = 100.0 / min(np.median(np.diff(arr_ref)), np.median(np.diff(arr_sig)))
    else:
        sps_float = float(sps)
     
    if dist_max is None:
        dist_max_float = min(np.median(np.diff(arr_ref)), np.median(np.diff(arr_sig))) / 4.0
    else:
        assert dist_max > 0.0
        dist_max_float = float(dist_max)
    
    # If symmetric Nearest Advocate, call the algorithm recursively
    if symmetric:
        left_shifts = _nearest_advocate(
            arr_ref=arr_ref_, arr_sig=arr_sig_, 
            td_min=float(td_min), td_max=float(td_max), dist_max=dist_max_float,
            sps=sps_float, sparse_factor=sparse_factor
            )
        right_shifts = _nearest_advocate(
            arr_ref=-arr_sig_[::-1], arr_sig=-arr_ref_[::-1], 
            td_min=float(td_min), td_max=float(td_max), dist_max=dist_max_float, 
            sps=sps_float, sparse_factor=sparse_factor
        )
        time_shifts = np.empty(left_shifts.shape, dtype=np.float32)
        time_shifts[:,0] = left_shifts[:,0]
        time_shifts[:,1] = (left_shifts[:,1] + right_shifts[:,1]) / 2
        return time_shifts
    
    return _nearest_advocate(
        arr_ref=arr_ref_, arr_sig=arr_sig_, 
        td_min=float(td_min), td_max=float(td_max), dist_max=dist_max_float,
        sps=sps_float, sparse_factor=sparse_factor
    )


if __name__ == "__main__":
    print("\nTesting Nearest Advocate for a single shift:")
    SIZE = 100
    np.random.seed(0)
    arr_reference = np.cumsum(np.random.random(size=SIZE) + 0.5)
    arr_signal = arr_reference + np.random.normal(loc=0, scale=0.1, size=SIZE) + np.pi
    print(nearest_advocate_single(arr_reference, arr_signal, dist_max=0.25))

    print("\nTesting Nearest Advocate for a search space:")
    print(nearest_advocate(
        arr_reference, arr_signal, dist_max=None,
        td_min=-10, td_max=10, sparse_factor=1, sps=None, symmetric=True
    )[:5])
