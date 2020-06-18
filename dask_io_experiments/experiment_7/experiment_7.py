import numpy as np
import os
import h5py
import sys
from cachey import nbytes
import time

def create_array(input_filepath, shape):
    arr = da.random.normal(size=shape)
    arr = arr.astype(np.float16)
    da.to_hdf5(input_filepath, '/data', arr, chunks=None, compression=None)
    with h5py.File(input_filepath, 'r') as f:
        print(f'Inspecting original array just created...')
        for k, v in f.items():
            print(f'\tFound object {v.name} at key {k}')
            if isinstance(v, Dataset):
                print(f'\t - Object type: dataset')
                print(f'\t - Physical chunks shape: {v.chunks}')
                print(f'\t - Compression: {v.compression}')
                print(f'\t - Shape: {v.shape}')
                print(f'\t - Size: {v.size}')
                print(f'\t - Dtype: {v.dtype}')


def split_to_hdf5(arr, f, nb_blocks=None):
    """ Split an array given its chunk shape. Output is a hdf5 file with as many datasets as chunks.

    Arguments:
    ----------
        arr: array to split
        f: an open hdf5 file to store data in it.
        nb_blocks: nb blocks we want to extract
    """
    arr_list = get_arr_chunks(arr, nb_blocks)
    datasets = list()

    for i, a in enumerate(arr_list):
        key = '/data' + str(i)
        print("creating chunk in hdf5 dataset -> dataset path: ", key)
        print("storing chunk of shape", a.shape)
        datasets.append(f.create_dataset(key, shape=a.shape))

    return da.store(arr_list, datasets, compute=False)


if __name__ == "__main__":
    paths = [
        "/home/tguedon/dask_io_experiments/",
        "/home/tguedon/dask_io",
        "/home/tguedon/dask",
        "/home/tguedon"
    ]
    for path in paths:
        sys.path.insert(0, path)

    print(sys.path)

    import dask
    import dask.array as da
    import dask_io
    from dask_io.optimizer.utils.utils import flush_cache
    from dask_io.optimizer.cases.case_creation import get_arr_chunks
    from dask.diagnostics import ResourceProfiler, Profiler, CacheProfiler, visualize
    from dask_io.optimizer.configure import enable_clustering, disable_clustering

    # arguments
    input_array_shape = (3850, 3025, 3500)
    split_cs = (770, 605, 700)
    input_directory = "/data/inputs"
    output_directory = "/data/outputs"
    ONE_GIG = 1000000000
    buffers_to_test = [3*ONE_GIG, 9*ONE_GIG]

    # create directories if do not exist
    for dirpath in [input_directory, output_directory]:
        if not os.path.isdir(dirpath):
            os.mkdir(dirpath)

    # remove output file if already exists
    input_array_name = "original_array.hdf5"
    output_array_name = "split_array.hdf5"
    input_filepath = os.path.join(input_directory, input_array_name)
    output_filepath = os.path.join(output_directory, output_array_name)
    if os.path.isfile(output_array_name):
        os.remove(output_array_name)

    # create input array
    if not os.path.isfile(input_filepath):
        print('Creating input array...')
        create_array(input_filepath, input_array_shape)

    # split
    times = list()
    for buffer in buffers_to_test:
        print("RUNNING BUFFER ", buffer)

        with h5py.File(input_filepath, 'r') as f_in: # open original array
            dset = f_in['/data']
            in_arr = da.from_array(dset, chunks=split_cs)

            with h5py.File(output_filepath, 'x') as f_out: # open split array
                # run optimized
                split_arr = split_to_hdf5(in_arr, f_out, nb_blocks=None)
                print("RUNNING OPTIMIZED")
                enable_clustering(buffer)
                flush_cache()
                with Profiler() as prof, ResourceProfiler() as rprof, CacheProfiler(metric=nbytes) as cprof:
                    with dask.config.set(scheduler='single-threaded'):
                        t = time.time()
                        _ = split_arr.compute()
                        t = time.time() - t
                        times.append([buffer, t, "optimized"])
                        visualize([prof, rprof, cprof], os.path.join(output_directory, str(buffer) + "opti" + ".html"), show=False)

            os.remove(output_filepath) # remove output file for next run
            with h5py.File(output_filepath, 'x') as f_out: # open split array
                # run non optimized
                split_arr = split_to_hdf5(in_arr, f_out, nb_blocks=None)
                print("RUNNING NON OPTIMIZED")
                disable_clustering()
                flush_cache()
                with Profiler() as prof, ResourceProfiler() as rprof, CacheProfiler(metric=nbytes) as cprof:
                    t = time.time()
                    _ = split_arr.compute()
                    t = time.time() - t
                    times.append([buffer, t, "non optimized"])
                    visualize([prof, rprof, cprof], os.path.join(output_directory, str(buffer) + "nonopti" + ".html"), show=False)

            os.remove(output_filepath) # remove output file for next run

    for r in times:
        print(r)



