import os, uuid, pdb, time, csv, itertools, traceback
from time import gmtime, strftime
import numpy as np
from random import shuffle

from dask_io_experiments.custom_setup import setup_all, EXP1_DIR
setup_all()

import dask
import dask.array as da
from dask.diagnostics import ResourceProfiler, Profiler, CacheProfiler, visualize
from cachey import nbytes

from dask_io.utils.utils import flush_cache, create_csv_file
from dask_io.main import enable_clustering, disable_clustering
from dask_io.utils.get_arrays import create_random_dask_array, save_to_hdf5
from dask_io.cases.case_validation import check_split_output_hdf5

from dask_io_experiments.test_config import TestConfig

from monitor.monitor import Monitor

def test_goodness_split(case_obj):
    disable_clustering()
    check_split_output_hdf5(case_obj.array_filepath, case_obj.out_filepath, case_obj.chunks_shape)

def run_to_hdf5(test):
    """ Execute a dask array with a given configuration.
    
    Arguments:
    ----------
        dask_config: contains the test configuration
    """
    flush_cache()
    if test.opti:
        enable_clustering(test.buffer_size)
    else:
        disable_clustering()

    try:
        arr = getattr(test, 'case').get()

        if test.opti:
            with dask.config.set(scheduler='single-threaded'):
                t = time.time()
                _ = arr.compute()
                t = time.time() - t
        else:
            t = time.time()
            _ = arr.compute()
            t = time.time() - t

        getattr(test, 'case').clean()  # close hdf5 file.
        return t

    except Exception as e:
        print(traceback.format_exc())
        print("An error occured during processing.")
        return False


def run_to_npy_stack(test):
    """ Execute a dask array with a given configuration.
    
    Arguments:
    ----------
        dask_config: contains the test configuration
    """
    flush_cache()
    if test.opti:
        enable_clustering(test.buffer_size)
    else:
        disable_clustering()

    a, b, c = getattr(test, 'case').get()

    try:
        with dask.config.set(scheduler='single-threaded'):
            t = time.time()
            _ = dask.base.compute_as_if_collection(a, b, c)
            t = time.time() - t
            return t

    except Exception as e:
        print(traceback.format_exc())
        print("An error occured during processing.")
        return False
    

def run_test(writer, test, output_dir):
    """ Wrapper around 'run' function for diagnostics.

    Arguments:
    ----------
        writer:
        test:
        output_dir:
    """
    with Profiler() as prof, ResourceProfiler() as rprof, CacheProfiler(metric=nbytes) as cprof:  
        uid = uuid.uuid4() 

        # monitor system resources
        log_filename = str(uid) + '.monitor.log'
        _monitor = Monitor(enable_print=False, enable_log=False, save_data=True)
        _monitor.disable_clearconsole()
        _monitor.set_delay(15)
        _monitor.start() 
        try:
            t = run_to_hdf5(test)
        finally:
            _monitor.stop()
            data = _monitor.get_pile()
            log_filepath = os.path.join(output_dir, log_filename)
            with open(log_filepath, 'w+') as logf:
                logf.writelines(data)

        # save dask diagnostics
        diagnostics_filename = str(uid) + '.html'
        print(f'Visualization file: {diagnostics_filename}')
        out_file_path = os.path.join(output_dir, diagnostics_filename)

        if t:
            visualize([prof, rprof, cprof], out_file_path)

        writer.writerow([
            getattr(test, 'hardware'), 
            getattr(test, 'cube_ref'),
            getattr(test, 'chunk_type'),
            getattr(test, 'chunks_shape'),
            getattr(test, 'opti'), 
            getattr(test, 'scheduler_opti'), 
            getattr(test, 'buffer_size'), 
            t,
            diagnostics_filename,
            uid 
        ])

        case = getattr(test, 'case')
        test_goodness_split(case)


def create_tests(options):
    """ Create all possible tests from a list of possible options.

    Arguments:
    ----------
        options: list of lists of configurations to try
    
    Returns 
    ----------
        A list of Test object containing the cartesian product of the combinations of "options"
    """
    def create_possible_tests(params):
        cube_type = params[1]
        chunk_type = params[3]
        test_list = list()
        for shape in chunks_shapes[cube_type][chunk_type]:
            if len(shape) != 3:
                print("Bad shape.")
                continue
            test_list.append(TestConfig((*params, shape)))
        return test_list

    tests_params = [e for e in itertools.product(*options)]
    tests = list()
    for params in tests_params:
        if len(params) == 6:
            tests = tests + create_possible_tests(params)

    if not len(tests) > 0:
        print("Tests creation failed.")
        exit(1)
    return tests
        

chunks_shapes = {
    "very_small":{
        "blocks":[(200, 200, 200)],
        "slabs":[(400, 400, 50)]
    },
    "small":{
        "blocks":[
            (700, 700, 700)],
        "slabs":[
            ("auto", 1400, 1400), 
            (5, 1400, 1400),
            (175, 1400, 1400)]
    },
    "big":{
        "blocks":[
            (350, 350, 350),
            (500, 500, 500),
            (875, 875, 875),],
            # (1750, 1750, 1750)], -> 1 block ne rentre pas en mémoire? refaire les calculs avec 2 bytes par valeur au lieu de 4
        "slabs":[
            # (3500, 3500, "auto"), -> dont know size
            # (3500, 3500, 1), -> trop gros graph
            (28, 3500, 3500),
            (50, 3500, 3500),]
            # (3500, 3500, 500)] -> 1 block ne rentre pas en mémoire? refaire les calculs avec 2 bytes par valeur au lieu de 4
    },
    "big_brain":{
        "blocks": [
            (770, 605, 700)
        ],
        "slabs": [
            
        ]
    }
}


def experiment(debug_mode,
    nb_repetitions,
    hardwares,
    cube_types,
    physical_chunked_options,
    chunk_types,
    scheduler_options,
    optimization_options):

    """ Apply the split algorithm using Dask arrays.

    Arguments:
    ----------
        nb_repetitions,
        hardwares,
        cube_types,
        physical_chunked_options,
        chunk_types,
        scheduler_options,
        optimization_options
        debug_mode: 
            False means need to run the tests (with repetitions etc.). 
            True means we will try the algorithm to see if the graph has been optimized as we wanted. 
            Dont run the actual tests, just the optimization.
    """
    
    output_dir = os.path.join(EXP1_DIR, 'outputs')
    out_filepath = os.path.join(output_dir, 'exp1_out.csv')

    print(f'Loading tests...')
    tests = create_tests([
        hardwares,
        cube_types,
        physical_chunked_options,
        chunk_types,
        scheduler_options,
        optimization_options,
    ])
    print(f'Done.')

    columns = ['hardware',
        'ref',
        'chunk_type',
        'chunks_shape',
        'opti',
        'scheduler_opti',
        'buffer_size', 
        'processing_time',
        'results_filepath'
    ]

    csv_path = os.path.join(output_dir, 'exp1_' + strftime("%Y-%m-%d %H:%M:%S", gmtime()) + '_out.csv')
    csv_out, writer = create_csv_file(csv_path, columns, delimiter=',', mode='w+')

    if not debug_mode: 
        tests *= nb_repetitions
        shuffle(tests)
        
    nb_tests = len(tests)
    for i, test in enumerate(tests):
        # create array file if needed
        if not os.path.isfile(getattr(test, "array_filepath")):
            try:
                print(f'Creating input array...')
                arr = create_random_dask_array(getattr(test, 'cube_shape'), distrib='uniform', dtype=np.float16)
                save_to_hdf5(arr, getattr(test, 'array_filepath'), physik_cs=getattr(test, 'physik_chunks_shape'), key='/data', compression=None)

            except Exception as e:
                print(traceback.format_exc())
                print("Input array creation failed.")
                continue

        print(f'\n\n[INFO] Processing test {i + 1}/{nb_tests} ~')
        test.print_config()
        run_test(writer, test, output_dir)
    csv_out.close()

