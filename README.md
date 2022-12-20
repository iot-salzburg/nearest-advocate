# nearest-advocate
A post-hoc time synchronization algorithm for event-based time-series data


## Setup

Within the GPU-Juypter-service mdilab-gupyter available on [dsws:22201](http://dsws:22201/), run:

```bash
cd /home/jovyan/work/Synchronization/
git clone https://github.com/iot-salzburg/nearest-advocate
cd nearest-advocate/src

pip install numpy
```


## Build and benchmark nearest_advocate

Build the algorith in `nearest_advocate` with:
```bash
cd nearest-advocate/src
python setup.py build_ext --inplace
```

To build and test the algorithm for one a search range or multiple time-shifts, run:

```bash
python run.py 
#>Numba:          2.23124337 s,    detected time shift: 3.15 s,    minimal mean distance: 0.079412 s
#>_Cython:        2.18392205 s,    detected time shift: 3.15 s,    minimal mean distance: 0.079412 s
#>Cython:         1.93009925 s,    detected time shift: 3.15 s,    minimal mean distance: 0.079412 s
```

Therefore, the Cython-version is even a little bit faster than numba.


## Run the tests

Run the testfile:

```bash
cd nearest-advocate/src
python tests/test_nearest_advocate.py
#> ok
```


## Development of Scipy

Read the the [build-README.md](#scipydev/REAMDE.md)

