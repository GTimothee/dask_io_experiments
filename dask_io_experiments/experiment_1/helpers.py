import os, json, sys, traceback, glob
import numpy as np
import time 
from time import gmtime, strftime


def clean_directory(datadir):
    workdir = os.getcwd()
    os.chdir(datadir)
    for filepath in glob.glob("[0-9]*_[0-9]*_[0-9]*.hdf5"):
        os.remove(filepath)
    os.chdir(workdir)
    


def write_monitor_logs(_monitor, uid, paths):
    data = _monitor.get_pile()
    log_filename = str(uid) + '.monitor.log'
    monitor_logfilepath = os.path.join(paths["outdir"], log_filename)
    with open(monitor_logfilepath, 'w+') as logf:
        logf.writelines(data)
    return monitor_logfilepath


def write_csv(rows, outdir, create_csv_file):
    csv_path = os.path.join(outdir, 'exp1_' + strftime("%Y-%m-%d %H:%M:%S", gmtime()) + '_out.csv')
    print(f'Creating csv file at {csv_path}.')
    csv_out, writer = create_csv_file(csv_path, columns, delimiter=',', mode='w+')
    for row in rows: 
      writer.writerow(row)
    csv_out.close()


def create_test_array(test, create_random_dask_array, save_to_hdf5):
    """ Create array file if needed.
    """
    if not os.path.isfile(getattr(test, "cuboid_filepath")):
        print(f'Creating input array...')
        try:
            params = getattr(test, 'params')
            path = getattr(test, 'cuboid_filepath')
            print(f'Creating file at {path}')
            t = time.time()
            arr = create_random_dask_array(params["array_shape"], distrib='uniform', dtype=np.float16)
            save_to_hdf5(arr, path, physik_cs=None, key='/data', compression=None)
            print(f'Time to create the input array: {time.time() - t}s')
        except Exception as e:
            print(traceback.format_exc())
            print("Input array creation failed.")
            return False
    else:
        print(f'Array already exists.')
    return True 


columns = ['hardware',
    'cuboid_name',
    'array_shape',
    'chunk_type',
    'chunk_shape',
    'optimized',
    'buffer_size', 
    'nthreads',
    'processing_time_split',
    'processing_time_merge',
    'diagnostics_split',
    'diagnostics_merge',
    'monitor_split',
    'monitor_merge',
    'success_run_split',
    'success_run_merge'
]

ONE_GIG = 1000000000

def test_goodness_split(splitcase):
    """ Only for split_hdf5, not split_multiple_hdf5
    """
    try:
        return check_split_output_hdf5(
            splitcase.array_filepath, 
            splitcase.out_filepath, 
            splitcase.chunks_shape)
    except:
        return None
