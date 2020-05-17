import os, uuid, pdb, time, csv, itertools, sys
import numpy as np
sys.path.insert(0, "./")

from random import shuffle
from cachey import nbytes
from time import gmtime, strftime

from dask_io_experiments.experiment_1.helpers import *
        

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

        if t:
            diagnostics = os.path.join(paths["outdir"], str(uid) + '.html')
            visualize([prof, rprof, cprof], diagnostics)   
        else:
            diagnostics = None
        return t, diagnostics, write_monitor_logs(_monitor, uid, paths)
    
        
def run_test(test, output_dir):
    """ Wrapper around 'run' function for diagnostics.

    Arguments:
    ----------
        test:
        output_dir:
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
    print(f'[Split] Find the diagnostics output file at {diagnostics_split}')
    print(f'[Split] Find the monitor output file at {monitor_split}')

    flush_cache()
    arr = mergecase.get()
    tmerge, diagnostics_merge, monitor_merge = run_to_hdf5(arr, params, uid)
    print(f'[Merge] Find the diagnostics output file at {diagnostics_merge}')
    print(f'[Merge] Find the monitor output file at {monitor_merge}')

    datadir = getattr(test, 'hardware_path')
    merge_filepath = getattr(test, 'merge_filepath')
    clean_directory(datadir, merge_filepath)

    test.clean_cases()  # close files.
    return [
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
        monitor_merge
    ]


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
        ["hdd", "ssd"],
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
        create_test_array(test, create_random_dask_array, save_to_hdf5)
        result = run_test(test, paths)
        results.append(result)

    write_csv(results, paths["outdir"])


if __name__ == "__main__":
    args = get_arguments()
    paths = load_config(args.config_filepath)
    custom_imports(paths)

    import dask
    import dask.array as da
    import dask_io
    from dask.diagnostics import ResourceProfiler, Profiler, CacheProfiler, visualize
    from dask_io.optimizer.utils.utils import flush_cache, create_csv_file
    from dask_io.optimizer.utils.get_arrays import create_random_dask_array, save_to_hdf5
    from dask_io.optimizer.cases.case_validation import check_split_output_hdf5
    from dask_io.optimizer.configure import enable_clustering, disable_clustering
    from dask_io_experiments.test_config import TestConfig
    from monitor.monitor import Monitor

    print("Output of monitor will be printed in 'outdir' if the run was successful.")
    print("Output csv file of experiment will be printed in 'outdir' even if a run failed.")


    """ Classic issues with shape selection:
    - 1 block ne rentre pas en mémoire
    - trop gros graph
    - dont know size if use `auto`
    """
    cuboids = {
        'test':{  
            'shape': (400, 400, 400),
            'buffer_size': 2 * ONE_GIG,
            'blocks':[(100, 100, 100)], # 64 blocks
            'slabs':[(5, 400, 400)] # 80 slabs
        },
        'small': {
            'shape':  (1400, 1400, 1400),
            'buffer_size': 5.5 * ONE_GIG,
            'blocks':[
                (700, 700, 700)],
            'slabs':[
                ("auto", 1400, 1400), 
                (5, 1400, 1400),
                (175, 1400, 1400)]
        },
        'big': {
            'shape': (3500, 3500, 3500),
            'buffer_size': 15 * ONE_GIG,
            'blocks':[
                (350, 350, 350),
                (500, 500, 500),
                (875, 875, 875),],
            'slabs':[
                (28, 3500, 3500),
                (50, 3500, 3500),]
        },
        'big_brain':{
            'shape': (3850, 3025, 3500),
            'buffer_size': 15 * ONE_GIG,
            'blocks': [
                (770, 605, 700) # 125
            ],
            'slabs': [
            ]
        }
    }
    if args.cuboids == None: 
        args.cuboids = list(cuboids.keys())
    if args.testmode:
        args.cuboids = ['test']

    experiment1()