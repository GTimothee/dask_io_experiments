import numpy as np
import csv


def get_buffers(R, B):
    buffers_partition = get_blocks_shape(R, B)
    buffers_volumes = get_named_volumes(buffers_partition, B)
    pass 


def read(buffer, in_dir):
    pass


def split_data(data):
    pass


def keep(data_to_keep, cache):
    pass


def write(data_to_write, out_dir):
    pass


def rechunk(in_dir, out_dir, R, I, O, B):
    buffers = get_buffers(R, B)
    cache = dict()
    t = time.time()
    for buffer in buffers:
        data = read(buffer, in_dir) 
        data_to_write, data_to_keep = split_data(data)
        keep(data_to_keep, cache)
        write(data_to_write, out_dir)
    return time.time() - t 


def get_arguments():
    """ Get arguments from console command.
    """
    parser = argparse.ArgumentParser(description="TODO")
    
    parser.add_argument('config_filepath', 
        action='store', 
        type=str, 
        help='Path to configuration file containing paths of third parties libraries, projects, data directories, etc. See README for more information.')

    parser.add_argument('-C', '--config_cases', 
        action='store',
        type=str,
        dest="config_cases",
        help='Path to configuration file containing cases. The default one is stored at dask_io_experiments/experiment_5/cases.json',
        default="./dask_io_experiments/experiment_5/cases.json")

    return parser.parse_args()


def init_dirs(paths):
    """ Create data directories and clean them if already exist
    """
    in_dir = os.path.join(paths["ssd_path"], "input_dir")
    out_dir = os.path.join(paths["ssd_path"], "output_dir")

    workdir = os.getcwd()
    for dirpath in [in_dir, out_dir]:
        if not os.path.isdir(dirpath):
            os.mkdir(dirpath) # create dir
        else:  # clean dir
            os.chdir(dirpath)
            for filename in glob.glob("*.hdf5"):
                os.remove(filename)
            os.chdir(workdir)
    return in_dir, out_dir


def clean_chunk_files(dirpath):
    """ Clean input and output files.
    """
    workdir = os.getcwd()
    os.chdir(dirpath)
    for filename in glob.glob("[0-9]*_[0-9]*_[0-9]*.hdf5"):
        os.remove(filename)
    os.chdir(workdir)


def create_input_files(in_dir, R, I):
    """ Randomly create input files for the rechunk process.
    """
    def get_filepath(in_volume, infiles_partition):
        _3d_pos = numeric_to_3d_pos(in_volume.index, infiles_partition, order='F')
        i, j, k = _3d_pos
        out_filename = f'{i}_{j}_{k}.hdf5'
        return os.path.join(in_dir, out_filename)

    infiles_partition = get_blocks_shape(R, I)
    infiles_volumes = get_named_volumes(infiles_partition, I)
    for in_volume in infiles_volumes:
        filepath = get_filepath(in_volume, infiles_partition)
        arr = create_random_dask_array(I, distrib='normal', dtype=np.float16)
        save_to_hdf5(arr, filepath, physik_cs=None, key='/data', compression=None)


def read_json(config_filepath):
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
    paths = read_json(args.config_filepath)
    custom_imports(paths)

    results = list()
    in_dir, out_dir = init_dirs(paths)

    for case in read_json(args.config_cases):  
        R, I, O, B, volumes_to_keep = case["R"], case["I"], case["O"], case["B"], case["volumestokeep"]
        create_input_files(in_dir, R, I)

        # process_time = rechunk(in_dir, out_dir, R, I, O, B)
        # results.append([
        #     hardware,
        #     R, 
        #     O, 
        #     I, 
        #     B,
        #     process_time
        # ])

        clean_chunk_files(in_dir)
        clean_chunk_files(out_dir)

    # write_csv(results, paths["outdir"])