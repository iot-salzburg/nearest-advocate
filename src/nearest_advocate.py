#!/usr/bin/env python
"""A time delay estimation method for event-based time-series."""

__author__ = "Christoph Schranz"
__copyright__ = "Copyright 2023, Salzburg Research"
__version__ = "0.1.9.1"
__maintainer__ = "Christoph Schranz, Mathias Schmoigl-Tonis"
__credits__ = ["Christoph Schranz", "Mathias Schmoigl-Tonis"]


from typing import Optional

import numpy as np

# Using the numba-implementation right now
from nearest_advocate_numba import nearest_advocate as _nearest_advocate
from nearest_advocate_numba import nearest_advocate_single as _nearest_advocate_single


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
    # cast the arrays to numpy arrays
    arr_ref_ = np.array(arr_ref, dtype=np.float32)
    arr_sig_ = np.array(arr_sig, dtype=np.float32)
    return _nearest_advocate_single(arr_ref=arr_ref_, arr_sig=arr_sig_,
                                    dist_max=dist_max,
                                    regulate_paddings=regulate_paddings, dist_padding=dist_padding)


def nearest_advocate(arr_ref: 'np.ndarray[np.float32]', arr_sig: 'np.ndarray[np.float32]',
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
    dist_max : float, optional
        Maximal accepted distances between two advocate events. Default None: 1/4 of the median gap of each array.
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

    Create a reference array whose events differences are sampled from a normal distribution. The signal array is the reference but shifted by `np.pi` and addional gaussian noise. The event-timestamps of both arrays must be sorted.

    >>> arr_ref = np.sort(np.cumsum(np.random.normal(loc=1, scale=0.25, size=1000)))
    >>> arr_sig = np.sort(arr_ref + np.pi + np.random.normal(loc=0, scale=0.1, size=1000))

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
    # cast the arrays to numpy arrays, subtract the minimal timestamp to avoid floating point error
    min_event_time = min(arr_ref[0], arr_sig[0])
    arr_ref_ = np.array(arr_ref - min_event_time, dtype=np.float32)
    arr_sig_ = np.array(arr_sig - min_event_time, dtype=np.float32)

    # Assert input properties
    assert len(arr_ref_.shape) == 1  # must be a 1D array
    assert len(arr_sig_.shape) == 1  # must be a 1D array
    assert arr_ref_.shape[0] > 1    # reference array must have >= 2 elements
    assert arr_sig_.shape[0] > 1    # signal array must have >= 2 elements
    assert isinstance(td_min, (int, float))
    assert isinstance(td_max, (int, float))
    assert isinstance(sparse_factor, int)
    assert isinstance(regulate_paddings, bool)
    return _nearest_advocate(arr_ref=arr_ref_, arr_sig=arr_sig_,
                             td_min=td_min, td_max=td_max, sps=sps,
                             sparse_factor=sparse_factor, dist_max=dist_max,
                             regulate_paddings=regulate_paddings, dist_padding=dist_padding)


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
