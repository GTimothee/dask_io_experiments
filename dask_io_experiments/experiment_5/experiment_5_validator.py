import random, sys, os, argparse, json, h5py, glob, math
import shutil, time
from time import gmtime, strftime
import numpy as np

def get_arguments():
    """ Get arguments from console command.
    """
    parser = argparse.ArgumentParser(description="This experiment is described as experiment 3 in Guédon et al. It is composed of three parts and tests dask_io.")
    
    parser.add_argument('config_filepath', 
        action='store', 
        type=str, 
        help='Path to configuration file containing paths of third parties libraries, projects, data directories, etc. See README for more information.')

    parser.add_argument('-n', '--nb_repetitions', action='store', 
        type=int, 
        dest='nb_repetitions',
        help='Number of repetitions for each case of the experiment. Default is 3.',
        default=3)

    parser.add_argument('-c', '--cases',
        action='store',
        type=list,
        dest='cases',
        help='List of cases indices to run. By default all cases are run. Use testmode (-t) to run only the "test" case. -t option overwrites this one.',
        default=None)

    parser.add_argument('-C', '--config_cases', 
        action='store',
        type=str,
        dest="config_cases",
        help='Path to configuration file containing cases. The default one is stored at dask_io_experiments/experiment_5/cases.json',
        default="./dask_io_experiments/experiment_5/cases.json")
    
    parser.add_argument('-t', '--testmode', 
        action='store_true', 
        dest='testmode',
        help='Test if setup working.',
        default=False)

    parser.add_argument('-o', '--overwrite', 
        action='store_true', 
        default=False,
        dest='overwritearray',
        help='Set to true to overwrite input array if already exists. Default is False.')

    return parser.parse_args()


def run_test_case(run):
    R, O, I, B, volumestokeep = tuple(run["R"]), tuple(run["O"]), tuple(run["I"]), tuple(run["B"]), run["volumestokeep"]
    print(f'Current run --- \nR: {R} \nO: {O} \nI: {I} \nB: {B} \n')
    d_arrays, d_regions = compute_zones(B, O, R, volumestokeep)

def run_case_1(run):
    def get_input_aggregate(O, I):
        lambd = list()
        dimensions = len(O)
        for dim in range(dimensions):
            lambd.append(math.ceil(O[dim]/I[dim])*I[dim])
        return lambd

    R, O, I = tuple(run["R"]), tuple(run["O"]), tuple(run["I"])
    lambd = get_input_aggregate(O, I)
    memorycases = [
        # [(1,1,lambd[2]), [1]],
        # [(1,lambd[1],lambd[2]), [1,2,3]],
        [(lambd[0],lambd[1],lambd[2]), list(range(1,8))]
    ]

    random.shuffle(memorycases)
    for memorycase in memorycases:
        print(f'Current run ---> R: {R}, O: {O}, I: {I}')
        print(f'Input aggregate shape: {lambd}')
        print(f'B: {memorycase}')
        B, volumestokeep = memorycase    
        
        remainders = [R[0]%B[0], R[1]%B[1], R[2]%B[2]]
        if not all(r == 0 for r in remainders):
            print(f"B does not define a partition of R, modify run in config file... Aborting.")
            print(f'Partition: {R[0]/B[0]}, {R[1]/B[1]}, {R[2]/B[2]}')
            print(f'Remainders: {R[0]%B[0]}, {R[1]%B[1]}, {R[2]%B[2]}')
            continue

        d_arrays, d_regions = compute_zones(B, O, R, volumestokeep)


def load_config(config_filepath):
    with open(config_filepath) as f:
        return json.load(f)

def custom_imports(paths):
    def isempty(s):
        if s == "":
            return True 
        return False 

    for k, path in paths.items():
        if "lib_" in k and not isempty(path):
            sys.path.insert(0, path)

if __name__ == "__main__":
    args = get_arguments()
    paths = load_config(args.config_filepath)
    custom_imports(paths)  # adding third-party libraries paths to the PYTHONPATH

    import dask
    import dask.array as da
    import dask_io
    from dask.diagnostics import ResourceProfiler, Profiler, CacheProfiler, visualize
    from dask_io.optimizer.utils.utils import flush_cache, create_csv_file, numeric_to_3d_pos
    from dask_io.optimizer.utils.get_arrays import create_random_dask_array, save_to_hdf5, get_dask_array_from_hdf5, clean_files
    from dask_io.optimizer.utils.array_utils import inspect_h5py_file
    from dask_io.optimizer.cases.case_validation import check_split_output_hdf5
    from dask_io.optimizer.configure import enable_clustering, disable_clustering
    from dask_io.optimizer.cases.case_config import Split, Merge
    from dask_io.optimizer.cases.resplit_case import compute_zones
    from dask_io.optimizer.cases.resplit_utils import get_blocks_shape
    from dask_io_experiments.experiment_5.helper import *

    import logging
    import logging.config
    logging.config.dictConfig({
        'version': 1,
        'disable_existing_loggers': True,
    })

    cases = load_config(args.config_cases)
    cases_to_run = get_cases_to_run(args,cases)

    for case_name, runs in cases.items():
        if case_name not in cases_to_run:
            continue 
        elif case_name == "case test":
            print(f'case test')
            execute_run = run_test_case
        elif case_name == "case 1":
            execute_run = run_case_1
            print(f'case 1')
        else:
            print("not supported yet")
            continue

        for run in runs:
            execute_run(run)