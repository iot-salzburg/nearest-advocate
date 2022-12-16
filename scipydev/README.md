# SciPy Dev-Documentation

Build nearest advocate in the SciPy Documenation.


## Setup

Within the GPU-Juypter-service mdilab-gupyter available on [dsws:22201](http://dsws:22201/) we follow the guides:

1. https://www.youtube.com/watch?v=K9bF7cjUJ7c and other of his videos
2. https://docs.scipy.org/doc/scipy/dev/contributor/cython.html#adding-cython


```bash
cd /home/jovyan/work/Synchronization/
git clone https://github.com/iot-salzburg/scipy.git

conda env create -f environment.yml
conda activate scipy-dev

git submodule update --init.
python dev.py build

# sudo apt-get install gfortran
# sudo python3 -m pip install -U pip
# conda install cython pythran pybind11 pooch numpy mkl=2019.* blas=*=*mkl

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


## Add nearest_advocate to scipy

Steps to add nearest_advocate to scipy:

2. nearest_advocate

```bash
cd /home/jovyan/work/Synchronization/scipy
git checkout -b nearest_advocate

cp /path/to/nearest_advocate_c.pyx scipy/signal/_nearest_advocate.pyx
```
Then change the files `setup.py`, `__init__.py` etc. as you can see in the directory scipydev

2. Build the documentation

```bash
conda install sphinx pydata-sphinx-theme sphinx-design matplotlib --channel conda-forge 
conda install setuptools wheel cython numpy pytest matplotlib conda-build
conda upgrade setuptools wheel cython numpy pytest matplotlib conda-build
pip install -r doc_requirements.txt
sudo apt install make

python setup.py build_ext --inplace
python runtests.py -v -t scipy.signal.tests.test_nearest_advocate

# set the working directory to the scipy development directory
conda develop .
# test the working directory, go into `scipy/doc` and call python -c "import scipy; print(scipy.__file__)"

cd doc
git submodule init
git submodule update

make html
# In case this doesn't work, run: make dist
```
