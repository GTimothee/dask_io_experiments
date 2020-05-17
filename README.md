# dask_io_experiments
Experiments and benchmarks for the dask_io module.

Description: 
- experiment 1: Benchmark of clustered implementation (split only)
- experiment 2: Evaluation of read/write time from hdf5 file to npy/hdf5 file(s)
- experiment 3: Test of resplit implementation
- experiment 4: Vanilla dask clean benchmark on split and merge tasks 

## Setup
Fill in the config file at dask_io_experiments/config.json
It is the same config file for all experiments.
The paths should be absolute to avoid any problem.
They are two types of paths.

The first type of paths is third-party libraries paths. 
If the path is let to "", then the program will assume you already installed it into your virtual environment.
These paths are here to use custom libraries from Github for example without installing it in a virtual environment.
To add a custom library, just add a new path in the config file with the key starting with "lib".
- lib_dask_io_path: path to dask_io library project
- lib_custom_dask_path: path to a custom dask library for use with dask_io
- lib_monitor_path: path to monitor, a light library to monitor the system consumption during an experiment execution (can be found on my Github)

The second type of paths is data paths:
- ssd_path: path to ssd directory to run the experiment and store temporary files
- hdd_path: path to hdd directory to run the experiment and store temporary files
- outdir: : path to directory to store the output file containing the results of the experiment

## Running experiments: 
Run the following commands from inside the "dask_io_experiments" main folder.
In any experiment, add "-t" to run it in test mode first and see if everything is OK before running the experiment on the big arrays.
Use "--help" or "-h" to see the command line arguments.

Experiment 4:
``` python dask_io_experiments/experiment_4/experiment_4.py --help ```
``` python dask_io_experiments/experiment_4/experiment_4.py -t "dask_io_experiments/pathsconfig.json" ```