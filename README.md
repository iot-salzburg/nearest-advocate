# Nearest-Advocate

A time delay estimation method for event-based time-series.

This repository contains the source code, demo data, research experiments as well as benchmarking of the code.


## Setup

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

## Run the tests

Run the testfile:

```bash
cd nearest-advocate/src
python tests/test_nearest_advocate.py
#> Testing numba-version:          ok
#> Testing Cython-version:         ok
#> Testing Python-version:         ok
```


## Build and benchmark nearest_advocate

To build and test the algorithm for one a search range or multiple time-shifts, run:

```bash
python run.py 
#>Numba:          0.01038599 s,    detected time shift: 3.15 s,    minimal mean distance: 0.076944 s
#>Cython:         0.01338649 s,    detected time shift: 3.15 s,    minimal mean distance: 0.076944 s
#>Python:         4.65694070 s,    detected time shift: 3.15 s,    minimal mean distance: 0.076944 s
```

Therefore, the Cython-version is even a little bit faster than the Numba-version.


## Reproduce the research experiments

In the `experiments` directory, there are multiple Jupyter Notebooks that contain experiements based on data in the `data` directory.


## Development of Scipy

Read the the [build-README.md](#scipydev/REAMDE.md)


## Citation 

When using in academic works please cite:

```
Christoph Schranz and Sebastian Mayr. 2022. Ein neuer Algorithmus zur Zeitsynchronisierung von Ereignis- basierten Zeitreihendaten als Alternative zur Kreuzkorrelation. (9 2022). https://doi.org/10.5281/ZENODO.7370958
```
