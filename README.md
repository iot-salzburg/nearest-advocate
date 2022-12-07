# nearest-advocate
A post-hoc time synchronization algorithm for event-based time-series data


## Setup

Within the GPU-Juypter-service mdilab-gupyter available on [dsws:22201](http://dsws:22201/) we follow the guides:

1. https://www.youtube.com/watch?v=K9bF7cjUJ7c
2. https://docs.scipy.org/doc/scipy/dev/contributor/cython.html#adding-cython


```bash
cd /home/jovyan/work/Synchronization/
git clone https://github.com/iot-salzburg/nearest-advocate
git clone https://github.com/scipy/scipy

sudo python3 -m pip install -U pip
pip install pybind11
pip install --upgrade cython  # the latest version is required
pip install pythran

cd scipy
python setup.py build_ext --inplace
```



## Run hello world

Create the file `scipy/scipy/optimize/mypython.py` with the input:
```python
def myfun():
    i = 1
    while i < 10000000:
        i += 1
    return i
```

Build the package again:
```bash
cd /home/jovyan/work/Synchronization/scipy
python setup.py build_ext --inplace
```

Now also create and initialize the Cython-file:
```bash
cp scipy/optimize/mypython.py scipy/optimize/mycython.pyx
```

Add in `setup.py` this line the appropriate section:
```python
    config.add_extension('mycython', sources=['mycython.c'])
```

Then create and call the file `scipy/hello_world.py`:
```python
import time
from scipy.optimize.mypython import myfun
start_time = time.time()
_ = myfun()
print(f"Python: \t{time.time() - start_time:6f}s")

from scipy.optimize.mycython import myfun
start_time = time.time()
_ = myfun()
print(f"Cython: \t{time.time() - start_time:6f}s")

from numba import njit
from scipy.optimize.mypython import myfun
mynbun = njit(myfun)
start_time = time.time()
_ = mynbun()
print(f"Numba:  \t{time.time() - start_time:6f}s")
```

Build the package again and run `hello_world.py`:

```bash
python setup.py build_ext --inplace
python hello_world.py 
# Python:         0.464833s
# Cython:         0.153764s
# Numba:          0.113668s
```
If the building fails with `mycython-checkpoint.pyx:1:0: 'mycython-checkpoint' is not a valid module name`, then delete all ipynb-checkfiles using this one-liner:

``` bash
find . -name .ipynb_checkpoints -exec rm {} \;
```


### Optimisation with static types

Change `mycython.pyx` to:
```python
def myfun():
    cdef int i = 1
```
Then build an run again:
```bash
python setup.py build_ext --inplace
python hello_world.py 
```


## Build and test nearest_advocate

Build the algorith in `nearest_advocate_c` with:
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

Therefore, t
