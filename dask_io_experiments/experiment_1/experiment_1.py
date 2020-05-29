import os, uuid, pdb, time, csv, itertools, sys, argparse, json
import numpy as np

from random import shuffle
from cachey import nbytes
        
def get_arguments():
    """
    arguments: 
        - config_filepath
    options: 
        - nb_repetitions
        - cuboids
        - testmode
    """
    parser = argparse.ArgumentParser(description="Split multidimensional arrays using vanilla dask and clustered strategy implementation from dask_io.")
    parser.add_argument('config_filepath', action='store', 
        type=str, 
        help='Path to configuration file containing paths of third parties libraries, projects, data directories, etc. See README for more information.')
    parser.add_argument('-n', '--nb_repetitions', action='store', 
        type=int, 
        help='Number of repetitions for each case of the experiment. Default is 3.',
        dest='nb_repetitions',
        default=3)
    parser.add_argument('--nthreads_opti', action='store', 
        type=int, 
        help='Number of threads for use with dask local scheduler for optimized run. Default is 1.',
        dest='nthreads_opti',
        default=1)
    parser.add_argument('--nthreads_non_opti', action='store', 
        type=int, 
        help='Number of threads for use with dask local scheduler for NON optimized run. Default is None => chosen by dask.',
        dest='nthreads_non_opti',
        default=None)
    parser.add_argument('-c', '--cuboids', action='store', 
        type=list, 
        help='Cuboids to experiment with. Experiment processes all cuboids by default.',
        dest='cuboids',
        default=None)
    parser.add_argument('-t', '--testmode', action='store_true',
        dest='testmode',
        help='Test if setup working.',
        default=False)
    return parser.parse_args()


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
    sys.path.insert(0, './')


def verify_results_merge(input_array_path, merged_array_path):
    original_array = get_dask_array_from_hdf5(input_array_path, "/data")
    merged_array = get_dask_array_from_hdf5(merged_array_path, "/data")
    verify_task = da.allclose(original_array, merged_array)
    print("VERIFY TASK: ", verify_task)
    disable_clustering()
    _res = verify_task.compute()
    print("RESULT: ", _res)
    if _res == False:
        print("[Error] Rechunk failed")
    clean_files()
    return _res


def verify_results_split(R, I, input_array_path, datadir):
    from dask_io.optimizer.cases.resplit_utils import get_blocks_shape
    splitfiles_partition = get_blocks_shape(R, I)
    print("split files partiton:", splitfiles_partition)

    all_true = True
    orig_arr = get_dask_array_from_hdf5(input_array_path, "/data", logic_cs=tuple(I))

    for i in range(splitfiles_partition[0]):
        for j in range(splitfiles_partition[1]):
            for k in range(splitfiles_partition[2]):
                splitfilename = f"{i}_{j}_{k}.hdf5"
                split_filepath = os.path.join(datadir, splitfilename)
                print("opening", split_filepath)
                splitarray = get_dask_array_from_hdf5(split_filepath, "/data")
                print(f"Slices from ground truth {i*I[0]}:{(i+1)*I[0]}, {j*I[1]}:{(j+1)*I[1]}, {k*I[2]}:{(k+1)*I[2]}")
                ground_truth_arr = orig_arr[i*I[0]:(i+1)*I[0],j*I[1]:(j+1)*I[1],k*I[2]:(k+1)*I[2]]

                verify_task = da.allclose(ground_truth_arr, splitarray)
                print("VERIFY TASK: ", verify_task)
                disable_clustering()
                _res = verify_task.compute()
                print("RESULT: ", _res)
                if _res == False:
                    print(f"[Error] Split failed for {splitfilename}")
                    all_true = False
    
    clean_files()
    return all_true


def run_to_hdf5(arr, params, uid):
    """ Execute a dask array with a given configuration.
    
    Arguments:
    ----------
        dask_config: contains the test configuration
    """
    
    def _compute(arr):
        t = time.time()
        _ = arr.compute()
        return time.time() - t

    
    def _compute_arr(arr):
        if params["nthreads"] == 1:
            print(f'Using the `single-threaded` scheduler...')
            with dask.config.set(scheduler='single-threaded'):
                return _compute(arr)
        else:
            return _compute(arr)

    
    with Profiler() as prof, ResourceProfiler() as rprof, CacheProfiler(metric=nbytes) as cprof:  
        _monitor = Monitor(enable_print=False, enable_log=False, save_data=True)
        _monitor.disable_clearconsole()
        _monitor.set_delay(15)
        _monitor.start() 

        t = None
        try:
            t = _compute_arr(arr)  
        finally:
            _monitor.stop()

        if t != None:
            diagnostics = os.path.join(paths["outdir"], str(uid) + '.html')
            # visualize([prof, rprof, cprof], diagnostics)   
        else:
            diagnostics = None
        return t, diagnostics, write_monitor_logs(_monitor, uid, paths)
    
        
def run_test(test, paths):
    """ Wrapper around 'run' function for diagnostics.

    Arguments:
    ----------
        test:
        paths:
    """
    test.print_config()
    uid = uuid.uuid4() 
    print("Test ID is ", str(uid))

    params = getattr(test, 'params')
    splitcase = getattr(test, 'splitcase')
    mergecase = getattr(test, 'mergecase')

    if params["optimized"]:
        enable_clustering(params["buffer_size"])
    else:
        disable_clustering()

    flush_cache()
    arr = splitcase.get()
    tsplit, diagnostics_split, monitor_split = run_to_hdf5(arr, params, uid)
    splitcase.clean()
    R = cuboids[params["cuboid_name"]]['shape']
    I = splitcase.chunks_shape
    print(f'R: {R}')
    print(f'I: {I}')
    if not 'auto' in I:
        success_run_split = verify_results_split(R, I, getattr(test, 'cuboid_filepath'), getattr(test, 'hardware_path'))
    else:
        success_run_split = None
    print(f'[Split] Find the diagnostics output file at {diagnostics_split}')
    print(f'[Split] Find the monitor output file at {monitor_split}')

    flush_cache()
    arr = mergecase.get()
    tmerge, diagnostics_merge, monitor_merge = run_to_hdf5(arr, params, uid)
    mergecase.clean()
    success_run_merge = verify_results_merge(getattr(test, 'cuboid_filepath'), getattr(test, 'merge_filepath'))
    print(f'[Merge] Find the diagnostics output file at {diagnostics_merge}')
    print(f'[Merge] Find the monitor output file at {monitor_merge}')

    datadir = getattr(test, 'hardware_path')
    merge_filepath = getattr(test, 'merge_filepath')
    clean_directory(datadir, merge_filepath)

    sample_res = [
        params["hardware"], 
        params["cuboid_name"],
        params["array_shape"],
        params["chunk_type"],
        params["chunk_shape"],
        params["optimized"],
        params["buffer_size"],
        params["nthreads"],
        round(tsplit, 4),
        round(tmerge, 4),
        diagnostics_split, 
        diagnostics_merge,
        monitor_split,
        monitor_merge,
        success_run_split,
        success_run_merge
    ]
    print("-------------RESULT\n", sample_res)
    return sample_res


def create_tests():    
    def get_test(cuboid, cuboid_info, hardware, chunk_shape, chunk_type, opti_status):
        """ Create a Test object from arguments.
        """
        optimized = False
        if opti_status == "optimized":
            optimized = True

        params = {
            "cuboid_name": cuboid,
            "hardware": hardware,
            "optimized": optimized,
            "array_shape": cuboid_info["shape"],
            "buffer_size": cuboid_info["buffer_size"],
            "chunk_type": chunk_type,
            "chunk_shape": chunk_shape
        }
        if optimized: 
            params["nthreads"] = args.nthreads_opti
        else :
            params["nthreads"] = args.nthreads_non_opti

        return TestConfig(params, paths)

    # Generate all combinations of test parameters
    print(f'Generating tests...')
    options = [
        ["ssd"], # WARNING running on ssd only
        args.cuboids,
        ["optimized", "non_optimized"]
    ]
    cartesian_res = [e for e in itertools.product(*options)]

    # Create tests
    tests = list()
    for test_infos in cartesian_res:
        hardware, cuboid, optimized = test_infos

        cuboid_info = cuboids[cuboid]
        for cs in cuboid_info["blocks"]:
            tests.append(get_test(cuboid, cuboid_info, hardware, cs, "blocks", optimized))
        for cs in cuboid_info["slabs"]:
            tests.append(get_test(cuboid, cuboid_info, hardware, cs, "slabs", optimized))

    if not len(tests) > 0:
        print("Tests creation failed.")
        exit(1)

    print(f'Done.')
    return tests


def experiment1():
    """ Split multidimensional arrays using vanilla dask and clustered strategy implementation from dask_io.
    """
    tests = create_tests() * args.nb_repetitions
    shuffle(tests)

    results = list()
    for i, test in enumerate(tests):
        print(f'\n\nProcessing test {i + 1}/{len(tests)} ~')
        print(f'Creating test array if needed...')
        create_test_array(test, create_random_dask_array, save_to_hdf5)
        clean_files()
        print(f'Done. Running test...')
        result = run_test(test, paths)
        results.append(result)

    write_csv(results, paths["outdir"], create_csv_file)


if __name__ == "__main__":
    args = get_arguments()
    paths = load_config(args.config_filepath)
    custom_imports(paths)

    import dask
    import dask.array as da
    import dask_io
    from dask.diagnostics import ResourceProfiler, Profiler, CacheProfiler, visualize
    from dask_io.optimizer.utils.utils import flush_cache, create_csv_file
    from dask_io.optimizer.utils.get_arrays import create_random_dask_array, save_to_hdf5, get_dask_array_from_hdf5, clean_files
    from dask_io.optimizer.cases.case_validation import check_split_output_hdf5
    from dask_io.optimizer.configure import enable_clustering, disable_clustering
    from dask_io_experiments.test_config import TestConfig
    from dask_io_experiments.experiment_1.helpers import *
    from monitor.monitor import Monitor

    print("Output of monitor will be printed in 'outdir' if the run was successful.")
    print("Output csv file of experiment will be printed in 'outdir' even if a run failed.")


    """ Classic issues with shape selection:
    - 1 block ne rentre pas en mémoire
    - trop gros graph
    - dont know size if use `auto`
    """
    cuboids = {
        # 'test':{  
        #     'shape': (400, 400, 400),
        #     'buffer_size': 2 * ONE_GIG,
        #     'blocks':[(100, 100, 100)], # 64 blocks
        #     'slabs':[(5, 400, 400)] # 80 slabs
        # },
        # 'small': {
        #     'shape':  (1400, 1400, 1400),
        #     'buffer_size': 5.5 * ONE_GIG,
        #     'blocks':[
        #         (700, 700, 700)],
        #     'slabs':[
        #         ("auto", 1400, 1400), 
        #         (5, 1400, 1400),
        #         (175, 1400, 1400)]
        # },
        'big': {
            'shape': (3500, 3500, 3500),
            'buffer_size': 15 * ONE_GIG,
            'blocks':[
                # (350, 350, 350),
                # (500, 500, 500),
                (875, 875, 875),],
            'slabs':[
                # (28, 3500, 3500),
                (50, 3500, 3500),]
        },
        # 'big_brain':{
        #     'shape': (3850, 3025, 3500),
        #     'buffer_size': 15 * ONE_GIG,
        #     'blocks': [
        #         (770, 605, 700) # 125
        #     ],
        #     'slabs': [
        #     ]
        # }
    }

    for p in [paths["hdd_path"], paths["ssd_path"]]:
        for cuboid_name in ["test", "small", "big", "big_brain"]:
            fp = os.path.join(p, cuboid_name + ".hdf5")
            if os.path.isfile(fp):
                print(f'Removing file {fp}')
                os.remove(fp)
            else:
                print(f'Input file {fp} does not exist')

    if args.cuboids == None: 
        args.cuboids = list(cuboids.keys())
    if args.testmode:
        args.cuboids = ['test']

    experiment1()
