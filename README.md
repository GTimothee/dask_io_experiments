# dask_io_experiments
Experiments and benchmarks for the dask_io module.


## setup
add dask_io_experiments.custom_setup.py

``` 
import sys
EXP1_DIR = ""
HDD_PATH = ""
SSD_PATH = ""

def setup_all():
    setup_custom_dask()
    setup_dask_io_package()

def setup_custom_dask():
    custom_dask_path = "/home/tim/projects/dask_io_experiments/dask"
    sys.path.insert(0, custom_dask_path)

def setup_dask_io_package():
    dask_io_path = "/home/tim/projects/dask_io_experiments/dask_io"
    sys.path.insert(0, dask_io_path)
``` 