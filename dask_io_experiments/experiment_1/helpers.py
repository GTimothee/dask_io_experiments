import argparse, os, json, sys, traceback, glob
import numpy as np
from time import gmtime, strftime


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


def clean_directory(datadir, merged_filepath):
    workdir = os.getcwd()
    os.chdir(datadir)
    for filepath in glob.glob("[0-9]*_[0-9]*_[0-9]*.hdf5"):
        os.remove(filepath)
    os.chdir(workdir)
    os.remove(merged_filepath)


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
            arr = create_random_dask_array(params["array_shape"], distrib='uniform', dtype=np.float16)
            save_to_hdf5(arr, path, physik_cs=None, key='/data', compression=None)
        except Exception as e:
            print(traceback.format_exc())
            print("Input array creation failed.")
            return False 
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

ONE_GIG = 1000000

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