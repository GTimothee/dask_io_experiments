""" In this experiment we test vanilla dask to split and merge a multidimensional array stored in an hdf5 file.
"""

import numpy as np
import json
import argparse
import sys, os, time, glob


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


def get_arguments():
    parser = argparse.ArgumentParser(description="In this experiment we test vanilla dask to split and merge a multidimensional array stored in an hdf5 file.")
    parser.add_argument('config_filepath', action='store', 
        type=str, 
        help='Path to configuration file containing paths of third parties libraries, projects, data directories, etc. See README for more information.')
    parser.add_argument('-t', '--testmode', action='store_true', default=False,
        dest='testmode',
        help='Test if setup working.')
    parser.add_argument('-o', '--overwrite', action='store_true', default=False,
        dest='overwritearray',
        help='Set to true to overwrite input array if already exists. Default is False.')
    return parser.parse_args()


def create_test_array(datadir, testmode, filepath):
    print("Creating test array in data directory ", datadir)
    if testmode:
        shape = (50,50,50) 
    else:
        shape = (1925, 1512, 1750) # 1/8 of big brain size
    
    arr = create_random_dask_array(shape, distrib='uniform', dtype=np.float16)

    try:
        save_to_hdf5(arr, filepath, physik_cs=None, key='/data', compression=None)
    except:
        print("Something went wrong while creating the test array. Aborting.")
        sys.exit(1)


"""
Both chunk shapes are ~same size in voxels:
- good cs: 
    chunk type: slice (slab with depth =1)
    2646000 voxels/slice
    1925 slices
    chunks partition: (1925,1,1)
- bad cs: 
    chunk type: block
    2756250 voxels/block
    1848 blocks
    chunks partition: (11,12,14)
"""
chunk_shapes = {
    "good_cs": (1,1512,1750), 
    "bad_cs": (175,126,125), 
    "testshape1": (10,10,10),
    "testshape2": (25,25,25) 
}


def run(arr):
    t = time.time()
    arr.compute()
    return time.time() - t


def split(datadir, filepath, cs):
    print("Splitting...")
    splitcase = Split(filepath, chunk_shapes[cs])
    splitcase.split_hdf5_multiple(datadir, nb_blocks=None)
    arr = splitcase.get()
    try:
        tsplit = run(arr)
        splitcase.clean()
        return tsplit
    except Exception as e: 
        print(e, "\nOops something went wrong... Aborting.")
        sys.exit(1)


def merge(datadir):
    print("Merging...")
    out_filepath = os.path.join(datadir, "merged.hdf5")
    mergecase = Merge(out_filepath)
    mergecase.merge_hdf5_multiple(datadir, data_key='/data', store=True)
    arr = mergecase.get()
    try:
        tmerge = run(arr)
    except Exception as e: 
        print(e, "\nOops something went wrong... Aborting.")
        sys.exit(1)
    mergecase.clean()
    return tmerge, out_filepath


def clean_directory(datadir):
    workdir = os.getcwd()
    os.chdir(datadir)
    for filepath in glob.glob("[0-9]*_[0-9]*_[0-9]*.hdf5"):
        os.remove(filepath)
    os.chdir(workdir)


def save_to_csv(outdir, rows):
    dateinfo = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())
    outfilename = 'exp4_' + dateinfo + '_out.csv'
    csv_path = os.path.join(outdir, outfilename)
    print(f'Writing output csv file at {csv_path}...')

    columns = ['hardware',
               'chunks_shape',
               'split_time',
               'merge_time']
    try:
        csv_out, writer = create_csv_file(csv_path, columns, delimiter=',', mode='w+')
        for row in rows:
            writer.writerow(row)
        csv_out.close()
    except Exception as e: 
        print(e, "\nOops something went wrong... Aborting.")
        sys.exit(1)


if __name__ == "__main__":
    args = get_arguments()
    testmode = args.testmode
    if testmode: 
        print("Running in test mode.")
        shapes = ["testshape1", "testshape2"]
    else:
        print("Running experiment.")
        shapes = ["good_cs", "bad_cs"]

    paths = load_config(args.config_filepath)
    custom_imports(paths)

    from dask_io.optimizer.utils.utils import flush_cache, create_csv_file
    from dask_io.optimizer.utils.get_arrays import create_random_dask_array, save_to_hdf5
    from dask_io.optimizer.cases.case_config import Split, Merge

    rows = list()
    for datadir, dirtype in zip([paths['ssd_path'], paths['hdd_path']], ['SSD', 'HDD']):
        filepath = os.path.join(datadir, "inputfile.hdf5")
        if args.overwritearray:
            if os.path.isfile(filepath):
                os.remove(filepath)
        clean_directory(datadir)
        create_test_array(datadir, testmode, filepath)

        for cs in shapes:
            flush_cache()
            t1 = split(datadir, filepath, cs)
            flush_cache()
            t2, merged_filepath = merge(datadir)

            rows.append([dirtype, chunk_shapes[cs], t1, t2])
            print("time to split: ", t1, "seconds.")
            print("time to merge: ", t2, "seconds")
            
            clean_directory(datadir)
            os.remove(merged_filepath)
        
        os.remove(filepath)

    save_to_csv(paths['outdir'], rows)
    print(f'Done.')