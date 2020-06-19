import random, sys, os, argparse, json, h5py, glob, math
import shutil, time
from time import gmtime, strftime
import numpy as np
import random, sys, os, argparse, json, h5py, glob, math
import shutil, time
from time import gmtime, strftime
import numpy as np
ONE_GIG = 1000000000


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
        return ["case2 test"] # ["case test", "case2 test"]
    elif args.cases != None:
        cases_to_run = list()
        for i in args.cases:
            casename = "case " + i
            if casename in all_cases_names:
                cases_to_run.append(casename)
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

    csv_path = os.path.join(outdir, 'exp5_' + strftime("%Y-%m-%d %H:%M:%S", gmtime()) + '_out.csv')
    print(f'Creating csv file at {csv_path}.')
    csv_out, writer = create_csv_file(csv_path, columns, delimiter=',', mode='w+')
    for row in rows: 
      writer.writerow(row)
    csv_out.close()


def load_config(config_filepath):
    with open(config_filepath) as f:
        return json.load(f)

def custom_imports(paths):
    def isempty(s):
        if s == "":
            return True 
        return False 

    for k, path in paths.items():
        if "lib_" in k and not isempty(path) and not path == "lib_custom_dask_path":
            sys.path.insert(0, path)

def split(inputfilepath, I, datadir):
    """ Split the input array stored at inputfilepath into outputfiles with shape I into datadir.
    Arguments: 
    ----------
        inputfilepath: Path to the input file we want to split.
        I: Output file shape. Shape of a chunk inside each output file.
        datadir: Path to directory in which to store the output files.
    """
    print("[preprocessing] Splitting input array...")
    case = Split(inputfilepath, I)
    case.split_hdf5_multiple(datadir, nb_blocks=None) 
    arr = case.get()
    buffer_shape = ONE_GIG * 5
    # enable_clustering(buffer_shape)
    with dask.config.set(scheduler='single-threaded'):
        arr.compute()
    # disable_clustering()
    case.clean()
    print(f'Split done.')


def apply_store(B, O, R, volumestokeep, reconstructed_array):
    """ Apply store, using the keep strategy.
    """
    # creations of data for dask store function
    d_arrays, d_regions = compute_zones(B, O, R, volumestokeep)
    out_files = list() # to keep outfiles open during processing
    sources = list()
    targets = list()
    regions = list()
    for outfile_index in sorted(d_arrays.keys()):
        sliceslistoflist = d_arrays[outfile_index]
        
        # create file
        outfiles_partition = get_blocks_shape(R, O)
        _3d_pos = numeric_to_3d_pos(outfile_index, outfiles_partition, order='F')
        i, j, k = _3d_pos
        out_filename = f'{i}_{j}_{k}.hdf5'
        out_file = h5py.File(os.path.join(outdir_path, out_filename), 'w')
        out_files.append(out_file)
        
        # create dset
        dset = out_file.create_dataset('/data', shape=O, dtype=np.float16)
        
        for i, st in enumerate(sliceslistoflist):
            tmp_array = reconstructed_array[st[0], st[1], st[2]]
            # print("Volume to be stored shape: ", tmp_array.shape)
            reg = d_regions[outfile_index][i]
            tmp_array = tmp_array.rechunk(tmp_array.shape)
            
            sources.append(tmp_array)
            targets.append(dset)
            regions.append(reg)

    return da.store(sources, targets, regions=regions, compute=False), out_files


def rechunk_keep(indir_path, outdir_path, B, O, R, volumestokeep, rechunk_input, model):
    case = Merge('samplename')
    case.merge_hdf5_multiple(indir_path, store=False, rechunk_input=rechunk_input)
    reconstructed_array = case.get()
    print("Merged array: ", reconstructed_array)
    print("Logically rechunking merged array to buffer shape...")
    reconstructed_array = reconstructed_array.rechunk(B)
    print("Merged array, rechunk:", reconstructed_array)

    rechunk_task, out_files = apply_store(B, O, R, volumestokeep, reconstructed_array)
    # rechunk_task.visualize(filename="tmp_dir/test_graph_keep.png")
    # sys.exit()


    print(f'Running algorithm...')
    with Profiler() as prof, ResourceProfiler(dt=0.25) as rprof, CacheProfiler() as cprof:
        with dask.config.set(scheduler='single-threaded'):
            try:
                t = time.time()
                rechunk_task.compute()
                t = time.time() - t
                # visualize([prof, rprof, cprof])
            except Exception as e: 
                print(e, "\nSomething went wrong during graph execution.")
                t = None

        diagnostics = os.path.join(outdir_path, 'exp5_' + str(model) + '.html')
        visualize([prof, rprof, cprof], diagnostics, show=False)   
    case.clean()
    for f in out_files:
        f.close()
    return t


def rechunk_vanilla_dask(indir_path, outdir_path, nthreads, R, O, model):
    """ Rechunk using vanilla dask
    """
    in_arrays = load_input_files(indir_path)

    case = Merge('samplename')
    case.merge_hdf5_multiple(indir_path, store=False)
    reconstructed_array = case.get()

    out_files = list() # to keep outfiles open during processing
    sources = list()
    targets = list()
    outfiles_partition = get_blocks_shape(R, O)
    for i in range(outfiles_partition[0]):
        for j in range(outfiles_partition[1]):
            for k in range(outfiles_partition[2]):
                out_filename = f'{i}_{j}_{k}.hdf5'
                out_file = h5py.File(os.path.join(outdir_path, out_filename), 'w')
                dset = out_file.create_dataset('/data', shape=O, dtype=np.float16)

                tmp_array = reconstructed_array[i*O[0]: (i+1)*O[0], j*O[1]: (j+1)*O[1], k*O[2]: (k+1)*O[2]]
                print(f'{i*O[0]}: {(i+1)*O[0]}, {j*O[1]}: {(j+1)*O[1]}, {k*O[2]}: {(k+1)*O[2]}')

                out_files.append(out_file)
                sources.append(tmp_array)
                targets.append(dset)

    rechunk_task = da.store(sources, targets, compute=False)
    # rechunk_task.visualize(filename="tmp_dir/test_graph_vanilla.png")
    # sys.exit()

    with Profiler() as prof, ResourceProfiler(dt=0.25) as rprof, CacheProfiler() as cprof:
        scheduler = 'single-threaded' if nthreads == 1 else 'threads'

        with dask.config.set(scheduler=scheduler):
            try:
                t = time.time()
                rechunk_task.compute()
                t = time.time() - t
                # visualize([prof, rprof, cprof])
            except Exception as e: 
                print(e, "\nSomething went wrong during graph execution.")
                t = None

        diagnostics = os.path.join(outdir_path, 'exp5_' + str(model) + '.html')
        visualize([prof, rprof, cprof], diagnostics, show=False)  

    clean_files()

    for f in out_files:
        f.close()

    return t 


def rechunk(indir_path, outdir_path, model, B, O, I, R, volumestokeep, rechunk_input):
    """ Rechunk data chunks stored into datadir using a given model.
    """
    if model == "vanilla1":
        return rechunk_vanilla_dask(indir_path, outdir_path, 1, R, O,model)
    elif model == "vanillan":
        return rechunk_vanilla_dask(indir_path, outdir_path, None, R, O, model)
    elif model == "keep":
        return rechunk_keep(indir_path, outdir_path, B, O, R, volumestokeep, rechunk_input, model)
    else:  # use plain python 
        return rechunk_plain_python(indir_path, outdir_path, B, O, I, R, model)


def verify_results(outdir_path, original_array_path, R, O):
    from dask_io.optimizer.cases.resplit_utils import get_blocks_shape
    outfiles_partition = get_blocks_shape(R, O)

    all_true = True
    with h5py.File(original_array_path, 'r') as f:
        orig_arr = f["/data"]

        for i in range(outfiles_partition[0]):
            for j in range(outfiles_partition[1]):
                for k in range(outfiles_partition[2]):
                    outfilename = f"{i}_{j}_{k}.hdf5"
                    with h5py.File(os.path.join(outdir_path, outfilename), 'r') as f:
                        data_stored = f["/data"]
                        print(f"Slices from ground truth {i*O[0]}:{(i+1)*O[0]}, {j*O[1]}:{(j+1)*O[1]}, {k*O[2]}:{(k+1)*O[2]}")
                        ground_truth = orig_arr[i*O[0]:(i+1)*O[0],j*O[1]:(j+1)*O[1],k*O[2]:(k+1)*O[2]]

                        # print(data_stored[()])
                        # print(ground_truth)
                        try:
                            assert np.allclose(data_stored[()], ground_truth)
                            print(f"Good output file {outfilename}")
                        except:
                            print(f"Error: bad rechunking {outfilename}")
                            all_true = False  # do not return here to see all failures
    return all_true


def run_test_case(run, inputfilepath, indir_path, outdir_path, results, hardware, model):
    R, O, I, B, volumestokeep = tuple(run["R"]), tuple(run["O"]), tuple(run["I"]), tuple(run["B"]), run["volumestokeep"]
    execute(R,O,I,B,inputfilepath, indir_path, outdir_path, results, hardware, model, volumestokeep, None)


def run_case_2(run, inputfilepath, indir_path, outdir_path, results, hardware, model):
    R, O, I, B, volumestokeep = tuple(run["R"]), tuple(run["O"]), tuple(run["I"]), tuple(run["B"]), run["volumestokeep"]
    rechunk_input = (min(B[0], I[0]), min(B[1], I[1]), min(B[2], I[2]))
    execute(R,O,I,B,inputfilepath, indir_path, outdir_path, results, hardware, model, volumestokeep, rechunk_input=rechunk_input)


def run_case_1(run, inputfilepath, indir_path, outdir_path, results, hardware, model):
    def get_input_aggregate(O, I):
        lambd = list()
        dimensions = len(O)
        for dim in range(dimensions):
            lambd.append(math.ceil(O[dim]/I[dim])*I[dim])
        return lambd

    R, O, I = tuple(run["R"]), tuple(run["O"]), tuple(run["I"])
    print(f'Current run ---> R: {R}, O: {O}, I: {I}')
    lambd = get_input_aggregate(O, I)
    print(f'Input aggregate shape: {lambd}')

    memorycases = [
        #[(1,1,lambd[2]), [1]],
        #[(1,lambd[1],lambd[2]), [1,2,3]],
        [(lambd[0],lambd[1],lambd[2]), list(range(1,8))]
    ]
    print(f'Buffer shapes to test: {memorycases}')

    random.shuffle(memorycases)
    for memorycase in memorycases:
        print(f'Testing buffer with shape: {memorycase}')
        B, volumestokeep = memorycase    
        
        # sanity check
        # print(f'Partition: {R[0]/B[0]}, {R[1]/B[1]}, {R[2]/B[2]}')
        # print(f'Remainders: {R[0]%B[0]}, {R[1]%B[1]}, {R[2]%B[2]}')
        remainders = [R[0]%B[0], R[1]%B[1], R[2]%B[2]]
        if not all(r == 0 for r in remainders):
            print(f"B does not define a partition of R, modify run in config file... Aborting.")
            continue
        else:
            execute(R,O,I,B,inputfilepath, indir_path, outdir_path, results, hardware, model, volumestokeep, None)


def create_test_array(filepath, shape):
    """ Create input dask array if does not exist.

    Array infos: 
    ------------
    - no physical chunks
    - drawn from normal distribution.
    - Dataset key = /data
    - Dtype = float16
    """
    # disable_clustering()
    if not os.path.isfile(filepath):
        print("Creating input array for the experiment...")
        arr = create_random_dask_array(shape, distrib='normal', dtype=np.float16)
        save_to_hdf5(arr, filepath, physik_cs=None, key='/data', compression=None)
        print(f'Done.')
    else:
        print("[input array creation] Input file already exists. Did nothing.")


def execute(R,O,I,B,inputfilepath, indir_path, outdir_path, results, hardware, model, volumestokeep, rechunk_input):
    print(f'Starting execution...')
    print(f'R={R}, \n\tO={O}, \n\tI={I}, \n\tB={B}, \n\tvolumestokeep={volumestokeep}')

    if "test" in case_name:
        if os.path.isfile(inputfilepath):
            os.remove(inputfilepath)
    create_test_array(inputfilepath, R)  # if not already created
    split(inputfilepath, I, indir_path)  # initially split the input array

    print(f'Rechunking with model "{model}"...')
    # disable_clustering()
    flush_cache()
    try:
        t = rechunk(indir_path, outdir_path, model, B, O, I, R, volumestokeep, rechunk_input)
        print(f'Rechunk done.')
        print("Processing time: ", t, " seconds. \nVerifying results (sanity check)...")
        success_run = verify_results(outdir_path, inputfilepath, R, O)
        print(f'Done. Sanity check successful: {success_run}')

        results.append([
            hardware, 
            case_name,
            R, 
            O, 
            I, 
            B, 
            model,
            round(t, 4),
            success_run
        ])
    except Exception as e:
        print(e, '\nAn error occured during execution of case')
        results.append([
            hardware, 
            case_name,
            R, 
            O, 
            I, 
            B, 
            model,
            None,
            False
        ])
    finally:
        clean_directory(indir_path)
        clean_directory(outdir_path)

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

    parser.add_argument('-m', '--model',
        type=str,
        action='store',
        dest='model',
        default='keep')

    return parser.parse_args()


def get_cases(args):
    cases = load_config(args.config_cases)
    return cases, get_cases_to_run(args, cases)


if __name__ == "__main__":
    """ IMPORTANT: We assume that for all run on both ssd/hdd the same R is used. 
    Details:
    --------
        It allows to create only once the input array which is very time consumming to generate.
        We did not create the file at the beginning of the "datadir for loop" due to the special case of the "test" case.
        Test case is not supposed to be run with the other cases, this should be ensured by the cmd line arguments.

    Other assumptions: 
    ------------------
    - we assume every hdf5 file contains only one dataset containing a chunk
    - each dataset is accessible with the key: "/data" (we load and store datasets with key "/data"), see h5py for details about datasets
    - we only work with float16 datatypes 
    - we assume partition is perfect i.e. buffers create a partition of R (no remainders)
    """
    # TODO: use scheduler constraint and unithreading for split/merge
    # TODO: in split modify buffer size dynamically either with buffer = input file or by argument
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
    # from dask_io.optimizer.configure import enable_clustering, disable_clustering
    from dask_io.optimizer.cases.case_config import Split, Merge
    from dask_io.optimizer.cases.resplit_case import compute_zones
    from dask_io.optimizer.cases.resplit_utils import get_blocks_shape
    from dask_io_experiments.experiment_5.plain_python_model import rechunk_plain_python
    from dask_io_experiments.experiment_5.helper import *

    import logging
    import logging.config
    logging.config.dictConfig({
        'version': 1,
        'disable_existing_loggers': True,
    })

    cases, cases_to_run = get_cases(args)
    # models = ["keep"]  # "dask_vanilla_1thread", "plain_python",
    print("model to run:", args.model)
    
    results = list()
    for datadir, hardware in zip([paths["ssd_path"]], ['ssd']):
        print("Data directory: ", datadir)

        inputfilepath = os.path.join(datadir, "original_array.hdf5")
        if os.path.isfile(inputfilepath) and args.overwritearray:
            print(f'Removing input file before starting...')
            os.remove(inputfilepath)
            print(f'Done.')

        # create 2 directories in datadir
        print("Creating data directories...")
        indir_path = os.path.join(datadir, "input_files")
        outdir_path = os.path.join(datadir, "output_files")
        for dirpath in [indir_path, outdir_path]:
            if not os.path.isdir(dirpath):
                os.mkdir(dirpath)
        print("Done. Running experiment...")

        for case_name, runs in cases.items():
            if case_name not in cases_to_run:
                continue 
            elif case_name == "case test":
                execute_run = run_test_case
            elif case_name == "case2 test":
                execute_run = run_case_2
            elif case_name == "case 1":
                execute_run = run_case_1
            elif case_name == "case 2":
                execute_run = run_case_2
            elif case_name == "case 3":
                execute_run = run_case_2
            else:
                print("not supported yet")
                continue
            
            print("Running case ", case_name, ", with ", args.nb_repetitions, " repetitions...")
            runs *= args.nb_repetitions
            random.shuffle(runs)

            for run in runs: 
                print("Current run:\n", run)
                print(f'Cleaning split files directory before splitting.')
                clean_directory(indir_path)
                print(f'Done.')
                execute_run(run, inputfilepath, indir_path, outdir_path, results, hardware, args.model)

    write_csv(results, paths["outdir"])
                    