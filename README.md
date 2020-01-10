# dask_io_experiments
Experiments and benchmarks for the dask_io module.


## setup
add dask_io_experiments.custom_setup.py

``` 
import sys, os
EXP1_DIR = ""
HDD_PATH = ""
SSD_PATH = ""

WORKSPACE = ""
PROJECT = ""
def setup_all():
    paths = [
        os.path.join(WORKSPACE, 'dask'),
        os.path.join(WORKSPACE, PROJECT, 'dask_io'),
        os.path.join(WORKSPACE, PROJECT, 'monitor')
    ]
    
    for path in paths:
        sys.path.insert(0, path)
``` 

+ create `outputs` dir in experiments_1 and 2 
