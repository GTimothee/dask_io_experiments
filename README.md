# dask_io_experiments
Experiments and benchmarks for the dask_io module.

- experiment 1: Benchmark of clustered implementation (split only)
- experiment 2: Evaluation of read/write time from hdf5 file to npy/hdf5 file(s)
- experiment 3: Test of resplit implementation
- experiment 4: Vanilla dask clean benchmark on split and merge tasks 

## setup
Fill in the config file at dask_io_experiments/sample_config.json
It is the same config file for all experiments.
The paths should be absolute to avoid any problem.
They are two types of paths.

The first type of paths is third-party libraries paths. 
If the path is let to "", then the program will assume you already installed it into your virtual environment.
These paths are here to use custom libraries from Github for example without installing it in a virtual environment.
To add a custom library, just add a new path in the config file with the key starting with "lib".
- lib_dask_io_path: path to dask_io library project
- lib_custom_dask_path: path to a custom dask 
- lib_monitor_path: path to monitor (can be found on my Github)

The second type of paths is data paths:
- ssd_path: path to ssd directory to run the experiment and store temporary files
- hdd_path: path to hdd directory to run the experiment and store temporary files
- outdir: : path to directory to store the output file containing the results of the experiment