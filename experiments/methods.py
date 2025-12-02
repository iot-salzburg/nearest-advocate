# pip install dtw-python
import dtw
import numpy as np
from numba import njit
from collections import Counter
from scipy.signal import correlate, correlation_lags


# parameters for PCC
PCC_MODE = "same"
PCC_SAMPLES_PER_S = 1000        # interpolated inter-event intervals

# parameters for KCC
KERNEL_PRECISION = 0.001

# parameters for DTW (tradeoff-between accuracy and runtime)
DTW_STEPWIDTH = 0.5


def moving_average(arr: np.ndarray, width: int) -> np.ndarray:
    """
    Smooth an array using a moving average with edge padding.
 
    Parameters
    ----------
    arr : np.ndarray
        Input array to be smoothed.
    w : int
        Kernel window length (incremented if even).
 
    Returns
    -------
    ndarray
        Smoothed array with the same length as input.
    """
    assert isinstance(width, int) and width > 2
    if width % 2 == 0:
        width = width + 1  # assert width is odd
    return np.convolve(
        np.concatenate([arr[:int(width//2)], arr, arr[-int(width//2):]]),
        np.ones(width),
        'valid'
    ) / width

    
def modify_timeseries(arr: np.ndarray, offset: float=0, subselect_length=None, sigma: float=0.0, fraction: float=1.0, time_warp_scale=0.0):
    """Modify a event-based timeseries in order generate semi-simulated data.

    Parameters
    ----------
    arr: np.ndarray
        Array to modify
    offset: float
        Offset to shift the time-series
    subselect_length: None, int
        If given, number of subsequent events selected
    sigma: float
        Amount of noise, relative to the median difference of subsequent events
    fraction: float
        Fraction of events used
    time_warp_scale: float
        Warp the time linearly by this factor

    Returns
    -------
    arr_modified : np.ndarray
        Modified array
    """
    arr_mod = arr.copy()

    # shift the array
    arr_mod += offset

    # select part
    if subselect_length:
        start_idx = np.random.randint(0, max(1, len(arr_mod)-subselect_length))
        arr_mod = arr_mod[start_idx:start_idx+subselect_length]

    # add gaussian noise to the events
    arr_mod = arr_mod + np.random.normal(loc=0, scale=sigma*np.median(np.diff(arr)), size=len(arr_mod))

    # sort the array to maintain continuity
    arr_mod.sort()

    # select the fraction of events
    if fraction < 1.0:
        arr_mod = arr_mod[np.random.random(len(arr_mod))<fraction]

    # linear time-warping
    if time_warp_scale > 0.0:
        pivot = arr_mod[len(arr_mod)//2]
        # warp_scale = np.random.normal(loc=1, scale=time_warp_scale)
        if np.random.rand() > 0.5:
            warp_scale = 1 + time_warp_scale
        else:
            warp_scale = 1 - time_warp_scale
        arr_mod = (arr_mod-pivot) * warp_scale + pivot

    return arr_mod


@njit(parallel=False)
def nearest_advocate_single(arr_ref: np.ndarray, arr_sig: np.ndarray,
                            dist_max: float) -> float:
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
    assert l_arr_ref >= 1 and l_arr_sig >= 1

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


@njit(parallel=False)
def nearest_advocate(arr_ref: np.ndarray, arr_sig: np.ndarray,
                     td_min: float, td_max: float, td_prox=None, sps: float=10, sparse_factor: int=1,
                     dist_max: float=0.0):
    '''Calculates the synchronicity of two arrays of timestamps for a search space between td_min and td_max with a precision of 1/sps. The synchronicity is given by the mean of all minimal distances between each event in arr_sig and it's nearest advocate in arr_ref.
    arr_ref (np.array): Reference array or timestamps assumed to be correct
    arr_sig (np.array): Signal array of  timestamps, assumed to be shifted by an unknown constant time-delta
    td_min (float): lower bound of the search space for the time-shift
    td_max (float): upper bound of the search space for the time-shift
    sps (int): number of investigated time-shifts per second, should be higher than 10 times the number of median gap of arr_ref (default 10).
    sparse_factor (int): factor for the sparseness of arr_sig for the calculation, higher is faster but may be less accurate (default 1)
    dist_max (None, float): Maximal accepted distances, default None: 1/4 of the median gap of arr_ref
    '''
    # set the default values for dist_max relative if not set
    # TODO improve default value: min(np.median(np.diff(arr_sig)), np.median(np.diff(arr_ref))) / 4
    if dist_max <= 0.0:
        dist_max = np.median(np.diff(arr_ref))/4
    if td_prox is None:
        td_prox = 1.5 * np.median(np.diff(arr_ref))

    # Random subsample and create a copy of arr_sig once, as it could lead to problems otherwise
    if sparse_factor > 1:
        probe = arr_sig[sparse_factor//2::sparse_factor].copy()
    else:
        probe = arr_sig.copy()

    # Create an k x 2 matrix to store the investigated time-shifts and their respective mean distance
    broad_search_n_samples = int((td_max-td_min)*sps)
    fine_search_n_samples = int(2*10*td_prox*sps)   # fine search with factor 1/10 sampling
    time_delays = np.empty((broad_search_n_samples+fine_search_n_samples, 2), dtype=np.float32)
    time_delays[:broad_search_n_samples, 0] = np.arange(td_min, td_max-1e-12, 1/sps)[:broad_search_n_samples]
    # Calculate the mean distance for all time-shifts in the search space.
    # The shift with the lowest mean distance is the best fit for the time-shift
    idx = 0
    while idx < broad_search_n_samples:
        # calculate the nearest advocate criteria
        time_delays[idx,1] = nearest_advocate_single(
             arr_ref,
             probe-time_delays[idx,0],  # the signal array is shifted by a time-delta
             dist_max=dist_max)
        idx += 1

    # finesearch around the peak
    time_shift, min_mean_dist = time_delays[np.argmin(time_delays[:broad_search_n_samples,1])]
    time_delays[broad_search_n_samples:, 0] = np.arange(time_shift-td_prox, time_shift+td_prox-1e-12,
                                                        1/sps/10)[:fine_search_n_samples]
    # Calculate the mean distance for all time-shifts in the search space.
    while idx < time_delays.shape[0]:
        # calculate the nearest advocate criteria
        time_delays[idx,1] = nearest_advocate_single(
             arr_ref,
             probe-time_delays[idx,0],  # the signal array is shifted by a time-delta
             dist_max=dist_max)
        idx += 1
    return time_delays


def pearson_cc_direct(
    arr_ref_cont: np.ndarray, arr_sig_cont: np.ndarray,
    mode: str="same", method: str="auto", significant_area=10
        ):
    """Pearson Cross-correlation between two time-series.
    
    The results are normalized between -1 and 1
    arr_ref_cont (np.array): Reference time-series assumed to be correct
    arr_sig_cont (np.array): Signal time-series, assumed to be shifted by an unknown constant time-delta
    mode (str): mode of the scipy's cross-correlation method: 'same', 'full'
    method (str): mode of the scipy's cross-correlation method: 'auto', 'fft', 'direct'
    significant_area (float): parameter for the extraction of the significant peak in the CC-curve,
        Specifies the width around the largest differences for the peak detection.
    """
    stepwidth = 1/PCC_SAMPLES_PER_S

    # calculate the correlation
    corrs = correlate(arr_sig_cont, arr_ref_cont, mode=mode, method=method) / np.sqrt(
        (correlate(arr_ref_cont, arr_ref_cont, mode=mode, method=method)[int(len(arr_ref_cont)/2)]
         * correlate(arr_sig_cont, arr_sig_cont, mode=mode, method=method)[int(len(arr_sig_cont)/2)]))
    time_lags = stepwidth * correlation_lags(
        arr_ref_cont.size, arr_sig_cont.size, mode=mode)

    # normalize the correlations
    overlap = (stepwidth + time_lags.max() - np.abs(time_lags))/time_lags.max()
    corrs = corrs / (overlap + 1e0)
    time_delays = np.array([time_lags, corrs]).T

    if mode != "same":
        raise Exception("Not implemented yet")
    if mode == "full":
        # overlap above 50%
        time_delays = time_delays[overlap > 0.5]

    # find the peak by searching for the largest differences in the signal
    significant_peak = np.argmax(np.diff(time_delays[:,1]))
    time_delays_proximity = time_delays[(significant_peak-significant_area*PCC_SAMPLES_PER_S//2
                                     ):significant_peak+significant_area*PCC_SAMPLES_PER_S//2]
    if len(time_delays_proximity) > 0:
        peak_idx = np.argmax(time_delays_proximity[:,1])
        time_shift, metric = time_delays_proximity[peak_idx]
    else:
        time_shift, metric = time_delays[np.argmax(time_delays[:,1])]
    return time_shift, metric, time_delays

    
def pearson_cc(arr_ref: np.ndarray, arr_sig: np.ndarray,
               mode: str=PCC_MODE, method: str="auto",
              smooth_outliers=False, significant_area=10):
    """Pearson Cross-correlation between two event-based time-series.
    An inter-event interpolation is performed in order to create continuous
    signals. The results are normalized between -1 and 1
    arr_ref (np.array): Reference array or timestamps assumed to be correct
    arr_sig (np.array): Signal array of  timestamps, assumed to be shifted by an unknown constant time-delta
    mode (str): mode of the scipy's cross-correlation method: 'same', 'full'
    method (str): mode of the scipy's cross-correlation method: 'auto', 'fft', 'direct'
    smooth_outliers (bool): flag if to smooth small or very high gaps after the interpolation
    significant_area (float): parameter for the extraction of the significant peak in the CC-curve,
        Specifies the width around the largest differences for the peak detection.
    """
    # Interpolate on a constant sample rate
    stepwidth = 1/PCC_SAMPLES_PER_S
    time_vector = np.arange(int(max(arr_ref[0], arr_sig[0])),
                            int(min(arr_ref[-1], arr_sig[-1])),
                            step=stepwidth)
    arr_ref_cont = np.interp(time_vector, arr_ref[:-1], np.diff(arr_ref))
    arr_sig_cont = np.interp(time_vector, arr_sig[:-1], np.diff(arr_sig))
    # arr_ref_ones = np.ones(len(arr_ref_cont))
    # arr_sig_ones = np.ones(len(arr_sig_cont))
    n = len(arr_ref_cont)

    # Remove outliers
    if smooth_outliers:
        l_bound = 0.5 * np.mean(np.diff(arr_ref))
        u_bound = 2.0 * np.mean(np.diff(arr_ref))
        arr_ref_cont[arr_ref_cont < l_bound] = l_bound; arr_ref_cont[arr_ref_cont > u_bound] = u_bound;
        arr_sig_cont[arr_sig_cont < l_bound] = l_bound; arr_sig_cont[arr_sig_cont > u_bound] = u_bound;

    # calculte the correlation
    corrs = correlate(arr_sig_cont, arr_ref_cont, mode=mode, method=method) / np.sqrt(
        (correlate(arr_ref_cont, arr_ref_cont, mode=mode, method=method)[int(len(arr_ref_cont)/2)]
         * correlate(arr_sig_cont, arr_sig_cont, mode=mode, method=method)[int(len(arr_sig_cont)/2)]))
    # corrs_ones = correlate(arr_sig_ones, arr_ref_ones, mode=mode, method=method)
    # corrs_ones /= corrs_ones.max()
    time_lags = stepwidth * correlation_lags(
        arr_ref_cont.size, arr_sig_cont.size, mode=mode)

    # normalize the correlations
    overlap = (stepwidth + time_lags.max() - np.abs(time_lags))/time_lags.max()
    corrs = corrs / (overlap + 1e0)
    time_delays = np.array([time_lags, corrs]).T

    if mode != "same":
        raise Exception("Not implemented yet")
    if mode == "full":
        # overlap above 50%
        time_delays = time_delays[overlap > 0.5]

    # find the peak by searching for the largest differences in the signal
    significant_peak = np.argmax(np.diff(time_delays[:,1]))
    time_delays_proximity = time_delays[(significant_peak-significant_area*PCC_SAMPLES_PER_S//2
                                     ):significant_peak+significant_area*PCC_SAMPLES_PER_S//2]
    if len(time_delays_proximity) > 0:
        peak_idx = np.argmax(time_delays_proximity[:,1])
        time_shift, metric = time_delays_proximity[peak_idx]
    else:
        time_shift, metric = time_delays[np.argmax(time_delays[:,1])]
    return time_shift, metric, time_delays


@njit
def discretize_arrays(arr_ref, arr_sig):
    start_ts = min(arr_ref[0],arr_sig[0])
    stop_ts = max(arr_ref[-1],arr_sig[-1])
    arr_ref_onehot = np.zeros(int((stop_ts-start_ts+1)/KERNEL_PRECISION))
    arr_sig_onehot = np.zeros(int((stop_ts-start_ts+1)/KERNEL_PRECISION))
    idx = 0
    arr_ref_idx = 0
    arr_sig_idx = 0
    cum_time = start_ts - 1
    for idx in range(len(arr_ref_onehot)):
        if arr_ref_idx< len(arr_ref) and cum_time > arr_ref[arr_ref_idx]:
            arr_ref_onehot[idx] = 1
            arr_ref_idx += 1
        if arr_sig_idx< len(arr_sig) and cum_time > arr_sig[arr_sig_idx]:
            arr_sig_onehot[idx] = 1
            arr_sig_idx += 1
        cum_time += KERNEL_PRECISION
    return arr_ref_onehot, arr_sig_onehot

@njit
def discretize_array(arr_ref):
    arr_ref_onehot = np.zeros(int((arr_ref[-1]-arr_ref[0]+1)/KERNEL_PRECISION))
    idx = 0
    arr_ref_idx = 0
    cum_time = 0
    for idx in range(len(arr_ref_onehot)):
        if arr_ref_idx< len(arr_ref) and cum_time > arr_ref[arr_ref_idx]:
            arr_ref_onehot[idx] = 1
            arr_ref_idx += 1
        cum_time += KERNEL_PRECISION
    return arr_ref_onehot

def kernel_cc(arr_ref: np.ndarray, arr_sig: np.ndarray,
              kernel_width: float=0.5,
               mode: str="same", method: str="auto"):
    """Kernel Cross-correlation between two event-based time-series.
    An a kernel convolution is performed in order to create continuous
    signals. The results are normalized between -1 and 1
    arr_ref (np.array): Reference array or timestamps assumed to be correct
    arr_sig (np.array): Signal array of  timestamps, assumed to be shifted by an unknown constant time-delta
    mode (str): mode of the scipy's cross-correlation method: 'same', 'full'
    method (str): mode of the scipy's cross-correlation method: 'auto', 'fft', 'direct'
    kernel_width (float): Width of the kernel for the convolution of the event-arrays.
    """
    # Discretize
    arr_ref_onehot, arr_sig_onehot = discretize_arrays(arr_ref, arr_sig)

    # Convolve with a triangular kernel of width
    kernel = kernel_width-np.abs(np.arange(-kernel_width, kernel_width+1e-3, 2*KERNEL_PRECISION))
    # kernel = kernel / kernel.sum()
    arr_ref_cont = np.convolve(arr_ref_onehot, kernel, mode="same")
    arr_ref_cont /= arr_ref_cont.max()
    arr_sig_cont = np.convolve(arr_sig_onehot, kernel, mode="same")
    arr_sig_cont /= arr_sig_cont.max()

    # calculte the correlation
    corrs = correlate(arr_sig_cont, arr_ref_cont, mode=mode, method="fft")
    time_lags =  KERNEL_PRECISION * correlation_lags(
        arr_sig_cont.size, arr_ref_cont.size, mode=mode)

    time_delays = np.array([time_lags, corrs]).T
    time_shift, metric = time_delays[np.argmax(time_delays[:,1])]

    return time_shift, metric, time_delays


# for more information see also: https://cs.fit.edu/~pkc/papers/tdm04.pdf
def dynamic_linear_timewarping(arr_ref: np.ndarray, arr_sig: np.ndarray, step_pattern=dtw.asymmetric):
    """Uses fastdtw implementation to calculate optimal alignment
    between two time series arrays with equal length or non-equal length.
    The warp path distance is a measure of the difference between the two time
    series after they have been warped together, which is measured by the sum
    of the distances between each pair of points connected by the vertical lines.
    Implementation: distance, warped_path = fastDTW(x, y, radius), where:
    x = timeseries of length n
    y = timeseries of length m
    step_pattern (dtw object): default is dtw.asymmetric
    """
    # Interpolate on a constant sample rate
    time_vector = np.arange(int(max(arr_ref[0], arr_sig[0])),
                            int(min(arr_ref[-1], arr_sig[-1])),
                            step=DTW_STEPWIDTH)
    arr_ref_cont = np.interp(time_vector, arr_ref[:-1], np.diff(arr_ref))
    arr_sig_cont = np.interp(time_vector, arr_sig[:-1], np.diff(arr_sig))

    ## Display the warping curve, i.e. the alignment curve
    alignment = dtw.dtw(arr_sig_cont, arr_ref_cont,
                    keep_internals=True, step_pattern=step_pattern,
                    window_type=None, open_begin=True, open_end=True)

    # create dataframe_array and time_shift result
    time_shifts, element_shifts = [], []
    for i, (e_ref, e_sig) in enumerate(zip(alignment.index1, alignment.index2)):
        if e_ref != 0 and e_sig != 0 and e_ref < len(arr_ref_cont)-1 and e_sig < len(arr_sig_cont)-1:
            element_shifts.append(e_ref - e_sig)
            # time_shift.append(time_vector[e_ref] - time_vector[e_sig])
            # print(e_ref, e_sig, time_vector[e_ref], time_vector[e_sig])

    # calculate modus return parameter
    try:
        element_shift_modus = Counter(element_shifts).most_common(1)[0][0]
        time_shift = element_shift_modus * DTW_STEPWIDTH
    except IndexError:
        if len(element_shifts) > 0:
            element_shift_modus = np.median(element_shifts)
            time_shift = element_shift_modus * DTW_STEPWIDTH
        else:
            time_shift = np.nan
    return time_shift, alignment
