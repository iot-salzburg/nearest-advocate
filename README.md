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


## Build and test nearest_advocate

Build the algorith in `nearest_advocate` with:
```bash
python setup.py build_ext --inplace
```

To build test the algorithm for one time-shift, run:

```bash
python tester_single.py 
#> Numba:          0.00054264 s,   mean_dist: 0.185051 s
#> Cython:         0.00045919 s,   mean_dist: 0.185051 s
```

To build and test the algorithm for one a search range or multiple time-shifts, run:

```bash
python tester_search.py 
#> Numba:          1.27184391 s,   mean_dist: [3.15       0.07923757] s
#> Cython:         1.06847000 s,   mean_dist: [3.15       0.07923757] s
```

Therefore, the Cython-version is a little bit faster than numba.


## Development of Scipy

Read the the [build-README.md](#scipydev/REAMDE.md)

