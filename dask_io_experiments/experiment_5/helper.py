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


def load_input_files(input_dirpath, dataset_key='/data'):
    """ Load input files created from the split preprocessing.
    """
    workdir = os.getcwd()
    os.chdir(input_dirpath)
    data = dict()
    for filename in glob.glob("[0-9]*_[0-9]*_[0-9]*.hdf5"):
        pos = filename.split('_')
        pos[-1] = pos[-1].split('.')[0]
        pos = tuple(list(map(lambda s: int(s), pos)))
        arr = get_dask_array_from_hdf5(filename, 
                                       dataset_key, 
                                       logic_cs="dataset_shape")
        data[pos] = arr
    os.chdir(workdir)
    return data


def get_cases_to_run(args, cases):
    all_cases_names = list(cases.keys())
    if args.testmode:
        return ["case test"]
    elif args.cases != None:
        cases_to_run = list()
        for i in args.cases:
            if isinstance(i, int):
                casename = "case " + str(i)
                if casename in all_cases_names:
                    cases_to_run.append(casename)
            else:
                print("Cases selected by command line should be integers.")
        if len(cases_to_run) == 0:
            raise ValueError("No case to run. Aborting.")
            sys.exit(1)
        return cases_to_run 
    else:
        return ["case 1", "case 2", "case 3"]


def clean_directory(dirpath):
    """ Remove intermediary files from split/rechunk.
    """
    workdir = os.getcwd()
    os.chdir(dirpath)
    for filename in glob.glob("[0-9]*_[0-9]*_[0-9]*.hdf5"):
        os.remove(filename)
    os.chdir(workdir)


def inspect_dir(dirpath):
    print(f'Inspecting {dirpath}...')
    workdir = os.getcwd()
    os.chdir(dirpath)
    nb_outfiles = 0
    for filename in glob.glob("[0-9]*_[0-9]*_[0-9]*.hdf5"):
        with h5py.File(os.path.join(dirpath, filename), 'r') as f:
            inspect_h5py_file(f)
        nb_outfiles += 1
    os.chdir(workdir)
    print(f'Found {nb_outfiles} files.')


def load_config(config_filepath):
    with open(config_filepath) as f:
        return json.load(f)


def write_csv(rows, outdir):
    columns = [
        'hardware',
        'case_name',
        'R',
        'O',
        'I',
        'B',
        'model', 
        'process_time',
        'sucess_run'
    ]

    csv_path = os.path.join(outdir, 'exp1_' + strftime("%Y-%m-%d %H:%M:%S", gmtime()) + '_out.csv')
    print(f'Creating csv file at {csv_path}.')
    csv_out, writer = create_csv_file(csv_path, columns, delimiter=',', mode='w+')
    for row in rows: 
      writer.writerow(row)
    csv_out.close()


def create_test_array(filepath, shape):
    """ Create input dask array if does not exist.

    Array infos: 
    ------------
    - no physical chunks
    - drawn from normal distribution.
    - Dataset key = /data
    - Dtype = float16
    """
    disable_clustering()
    if not os.path.isfile(filepath):
        print("Creating input array for the experiment...")
        arr = create_random_dask_array(shape, distrib='normal', dtype=np.float16)
        save_to_hdf5(arr, filepath, physik_cs=None, key='/data', compression=None)
        print(f'Done.')
    else:
        print("[input array creation] Input file already exists. Did nothing.")



def custom_imports(paths):
    def isempty(s):
        if s == "":
            return True 
        return False 

    for k, path in paths.items():
        if "lib_" in k and not isempty(path):
            sys.path.insert(0, path)