# Nearest-Advocate

A time delay estimation method for event-based time-series.

This repository contains the source code, demo data, research experiments as well as benchmarking of the code and is based on the Github-Repository [iot-salzburg/nearest-advocate](https://github.com/iot-salzburg/nearest-advocate).


## Quickstart

Install the package with:

```bash
pip install nearest_advocate
```

Open Python and mport and use it for time delay estimation of event-based time-series:

```python
import numpy as np
import nearest_advocate
```  

Create a reference array whose inter-event intervals are sampled from a normal distribution. The signal array is a clone of the reference´, shifted by `np.pi` and added Gaussian noise. The event's timestamps of both arrays must be sorted.

```python
arr_ref = np.sort(np.cumsum(np.random.normal(loc=1, scale=0.25, size=1000)))
arr_sig = np.sort(arr_ref + np.pi + np.random.normal(loc=0, scale=0.1, size=1000))
```

The function `nearest_advocate.nearest_advocate` returns a two-columned array with all investigated time-shifts and their mean distances, i.e., the measure of the synchronicity between both array (lower is better). 

```python
time_shifts = nearest_advocate.nearest_advocate(arr_ref=arr_ref, arr_sig=arr_sig, td_min=-60, td_max=60, sps=10)
time_shift, min_mean_dist = time_shifts[np.argmin(time_shifts[:,1])]
print(f"Found an optimum at {time_shift:.4f}s with a minimal mean distance of {min_mean_dist:.6f}s")
#> Found an optimum at 3.15s with a minimal mean distance of 0.079508s
```

Create a plot of the resulting characteristic curve

```python 
import matplotlib.pyplot as plt
plt.plot(time_shifts[:,0], time_shifts[:,1], color="steelblue", label="Mean distance")
plt.vlines(x=time_shift, ymin=0.05, ymax=0.25, color="firebrick", label=f"Shift = {time_shift:.2f}s")
plt.xlim(0, 8)
plt.xlabel("Time delay (s)")
plt.ylabel("Mean distance (s)")
plt.legend(loc="lower right")
plt.savefig("tmp.png")
plt.show()
```


## Building from source

### Setup

```bash
cd /go/to/path
git clone https://github.com/iot-salzburg/nearest-advocate
cd nearest-advocate
pip install -r requirements.txt
```

For reproducibility, the experiments were run in Python 3.9 inside a container environment using the Docker orchestration software with the image `cschranz/gpu-jupyter` and tag `v1.4\_cuda-11.0\_ubuntu-20.04`, available on Dockerhub.

Build the Cython-version of the algorithm with:

```bash
cd src
python setup.py build_ext --inplace
```


### Run the tests

Currently, the Cython-version is under development and will be available soon.

Run the testfile:

```bash
cd nearest-advocate
python tests/test_algorithm.py
#> Testing numba-version:          ok
#> Testing Cython-version:         ok
#> Testing Python-version:         ok

python tests/test_performances.py
#> ################# Test and compare shifts ##################
#> Numba:          0.01329827 s,    detected time shift: 3.15 s,    minimal mean distance: 0.084238 s
#> Cython:         0.01338649 s,    detected time shift: 3.15 s,    minimal mean distance: 0.084238 s
#> Python:         3.06915808 s,    detected time shift: 3.15 s,    minimal mean distance: 0.084238 s
#> 
#> ########## Compare versions for multiple lengths ###########
#> Method      10       100       1000     10000     100000  
#> Numba:   0.000157  0.000786  0.013276  0.138520  1.402027 
```


## Reproduce the research experiments

In the directory [nearest_advocate/experiments](https://github.com/iot-salzburg/nearest-advocate/tree/main/experiments), there are multiple Jupyter Notebooks that contain experiements based on data in the `data` directory.


<!-- ## Development of Scipy

Read the the [build-README.md](#scipydev/REAMDE.md)
 -->


## Citation 

When using in academic works please cite:

```
Christoph Schranz and Sebastian Mayr. 2022. Ein neuer Algorithmus zur Zeitsynchronisierung von Ereignis- basierten Zeitreihendaten als Alternative zur Kreuzkorrelation. (9 2022). https://doi.org/10.5281/ZENODO.7370958
```
