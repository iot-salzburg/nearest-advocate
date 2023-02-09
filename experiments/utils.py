import numpy as np
from numba import njit


def moving_aggregation(x, w, aggregation="median"):
    # Performs a moving aggregation on an array
    # x: array
    # w: length of the kernel window 
    # aggregation: type of moving aggregation, 'median', 'min' or 'max'
    if aggregation == "median": agg_fct = np.median
    elif aggregation == "mean": agg_fct = np.mean
    elif aggregation == "min": agg_fct = np.min
    elif aggregation == "max": agg_fct = np.max
    else: raise f"Unknown aggregation {aggregation}, set one of 'median', 'mean', 'min', 'max'"
    
    shifted = np.zeros((len(x)+w-1, w))
    shifted[:,:] = np.nan
    for idx in range(w-1):
        shifted[idx:-w+idx+1, idx] = x
    shifted[idx+1:, idx+1] = x

    aggregated = agg_fct(shifted, axis=1)
    for idx in range(w-1):
        # TODO maybe from 0 to idx+1, analog for the last entries
        # raise "TODO"
        aggregated[idx] = agg_fct(shifted[idx, :idx+1])
        aggregated[-idx-1] = agg_fct(shifted[-idx-1, -idx-1:])
    return aggregated[(w-1)//2:-(w-1)//2]

@njit
def moving_aggregation_nb(x, w, aggregation="median"):
    # Performs a moving aggregation on an array
    # x: array
    # w: length of the kernel window 
    # aggregation: type of moving aggregation, 'median', 'min' or 'max'
    
    # build the shifted value block for the aggregation
    shifted = np.zeros((len(x)+w-1, w), dtype=np.float32)
    shifted[:,:] = np.nan
    for idx in range(w-1):
        shifted[idx:-w+idx+1, idx] = x
    shifted[idx+1:, idx+1] = x
    
    # switch cases as numba does not support the reference of compiled functions yet
    aggregated = np.zeros(shifted.shape[0], dtype=np.float32)
    if aggregation == "median": 
        for row_idx in range(shifted.shape[0]):
            aggregated[row_idx] = np.median(shifted[row_idx, :])
        for idx in range(w-1):
            aggregated[idx] = np.median(shifted[idx, :idx+1])
            aggregated[-idx-1] = np.median(shifted[-idx-1, -idx-1:])
        return aggregated[(w-1)//2:-(w-1)//2]
    elif aggregation == "mean": 
        for row_idx in range(shifted.shape[0]):
            aggregated[row_idx] = np.mean(shifted[row_idx, :])
        for idx in range(w-1):
            aggregated[idx] = np.mean(shifted[idx, :idx+1])
            aggregated[-idx-1] = np.mean(shifted[-idx-1, -idx-1:])
        return aggregated[(w-1)//2:-(w-1)//2]
    if aggregation == "min":
        for row_idx in range(shifted.shape[0]):
            aggregated[row_idx] = np.min(shifted[row_idx, :])
        for idx in range(w-1):
            aggregated[idx] = np.min(shifted[idx, :idx+1])
            aggregated[-idx-1] = np.min(shifted[-idx-1, -idx-1:])
        return aggregated[(w-1)//2:-(w-1)//2]
    elif aggregation == "max": 
        for row_idx in range(shifted.shape[0]):
            aggregated[row_idx] = np.max(shifted[row_idx, :])
        for idx in range(w-1):
            aggregated[idx] = np.max(shifted[idx, :idx+1])
            aggregated[-idx-1] = np.max(shifted[-idx-1, -idx-1:])
        return aggregated[(w-1)//2:-(w-1)//2]
    else: return None
    # else: raise f"Unknown aggregation {aggregation}, set one of 'median', 'mean', 'min', 'max'"
    

if __name__ == "__main__":
    print(f"\nTesting moving_aggregation with 'min':")
    print(moving_aggregation(np.arange(10), 4, aggregation="min"))
    
    print(f"\nTesting moving_aggregation_nb with 'min':")
    print(moving_aggregation_nb(np.arange(10), 4, aggregation="min"))
    