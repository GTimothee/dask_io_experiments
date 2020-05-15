""" In this experiment we test vanilla dask to split and merge a multidimensional array stored in an hdf5 file.
"""

import numpy as np
import argparse

arser = argparse.ArgumentParser()

parser.add_argument('datadir', metavar='N', type=str, nargs='+', help='Directory to put and manipulate data files')

from dask_io.utils.get_arrays import create_random_dask_array, save_to_hdf5
from dask_io.cases.case_config import Split, Merge
from dask_io.utils.utils import flush_cache


def create_test_array():
    shape = (1925, 1512, 1750) # 1/8 of big brain size
    filepath = os.path.join(datadir, "inputfile.hdf5")
    arr = create_random_dask_array(shape, distrib='uniform', dtype=np.float16)
    save_to_hdf5(arr, filepath, physik_cs=None, key='/data', compression=None)
    return filepath


chunk_shapes = {
    "good_cs": (),
    "bad_cs": () 
}


def run():
    t = time.time()
    arr.compute()
    return time.time() - t


def split(datadir, filepath, cs):
    splitcase = Split(filepath, chunk_shapes[cs])
    splitcase.split_hdf5_multiple(self, datadir, nb_blocks=None)
    arr = splitcase.get()
    tsplit = run(arr)
    splitcase.clean()
    return tsplit


def merge(datadir):
    out_filepath = os.path.join(datadir, "merged.hdf5")
    mergecase = Merge(out_filepath)
    mergecase.merge_hdf5_multiple(datadir, data_key='/data', store=True)
    arr = mergecase.get()
    tmerge = run(arr)
    mergecase.clean()
    return tmerge


if __name__ == "__main__":
    filepath = create_test_array()
    
    for cs in ["good_cs", "bad_cs"]:
        flush_cache()
        split(datadir, filepath, cs)
        flush_cache()
        merge(datadir)

        