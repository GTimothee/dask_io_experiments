import os
import h5py
import pytest
import numpy as np
import glob
import shutil
import tempfile
import sys

sys.path.insert(0, '.')

from dask_io_experiments.custom_setup import setup_all
setup_all()

import dask
import dask.array as da
from dask.diagnostics import Profiler, ResourceProfiler, CacheProfiler
from dask.diagnostics import visualize

from dask_io.optimizer.cases.case_config import Split, Merge
from dask_io.optimizer.cases.case_creation import get_arr_chunks
from dask_io.optimizer.configure import enable_clustering, disable_clustering
from dask_io.optimizer.utils.utils import ONE_GIG, CHUNK_SHAPES_EXP1
from dask_io.optimizer.utils.get_arrays import get_dask_array_from_hdf5
from dask_io.optimizer.utils.array_utils import inspect_h5py_file
from dask_io.optimizer.utils.get_arrays import create_random_dask_array, save_to_hdf5
from dask_io.optimizer.cases.resplit_case import compute_zones

def create_test_array_nochunk(file_path, shape):
    """ Create input dask array for the experiment with no physical chunks.
    """
    
    if not os.path.isfile(file_path):
        arr = create_random_dask_array(shape, distrib='normal', dtype=np.float16)
        save_to_hdf5(arr, file_path, physik_cs=None, key='/data', compression=None)


def split(inputfilepath, I):
    filetosplitpath = inputfilepath
    splitfilesshape = I
    case = Split(filetosplitpath, splitfilesshape)
    case.split_hdf5_multiple('./', nb_blocks=None) # split all blocks into different files
    arr = case.get()
    arr.compute()
    case.clean()


def use_temp_directory():
    tmpdir = tempfile.TemporaryDirectory()
    os.chdir(tmpdir.name)
    return tmpdir.name


if __name__ == "__main__":
    tmpdir = tempfile.TemporaryDirectory()
    os.chdir(tmpdir.name)

    # split -> prepare case
    buffer_size = 4 * ONE_GIG
    inputfilepath = './small_array_nochunk.hdf5'
    inputfileshape = (1,120,120)
    R = (1,120,120)
    I = (1,30,30)
    O = (1,40,40)
    B = (1,60,60)

    create_test_array_nochunk(inputfilepath, inputfileshape)
    split(inputfilepath, I)    

    # resplit
    case = Merge('./reconstructed.hdf5') # dont care about the name of outfile bec we retrieve without actually merging
    case.merge_hdf5_multiple('./', store=False)
    reconstructed_array = case.get()
    print(reconstructed_array)
    # reconstructed_array = reconstructed_array.rechunk(1,60,60)  # creation des noeuds de buffer
    # print(reconstructed_array)

    d_arrays, d_regions = compute_zones(B, O, R, [1])

    out_files = list() # to keep outfiles open during processing
    sources = list()
    targets = list()
    regions = list()
    for outfile_index in range(9):
        sliceslistoflist = d_arrays[outfile_index]
        
        # create file
        out_file = h5py.File('./' + str(outfile_index) + '.hdf5', 'w')
        out_files.append(out_file)
        
        # create dset
        dset = out_file.create_dataset('/data', shape=O)
        
        for i, st in enumerate(sliceslistoflist):
            tmp_array = reconstructed_array[st[0], st[1]]
            reg = d_regions[outfile_index][i]
            tmp_array = tmp_array.rechunk(tmp_array.shape)
            
            sources.append(tmp_array)
            targets.append(dset)
            regions.append(reg)

    dask.config.set({
        'optimizations': []
    })

    task = da.store(sources, targets, regions=regions, compute=False)
    
    # compute / print graph / show results
    with Profiler() as prof, ResourceProfiler(dt=0.25) as rprof, CacheProfiler() as cprof:
        with dask.config.set(scheduler='single-threaded'):
            task.compute()
        visualize([prof, rprof, cprof])

    sys.exit()

    outfiles = list()
    for fpath in glob.glob("[0-9].hdf5"):  # remove split files from previous tests
        f = h5py.File(fpath, 'r')
        print(f'Filename: {fpath}')
        inspect_h5py_file(f)

    task.visualize(optimize_graph=False, filename='/tmp/viz.svg')