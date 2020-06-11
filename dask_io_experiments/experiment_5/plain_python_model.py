import os, glob, h5py
import numpy as np
import time

def get_input_files(input_dirpath):
    workdir = os.getcwd()
    os.chdir(input_dirpath)
    infiles = list()
    for filename in glob.glob("[0-9]*_[0-9]*_[0-9]*.hdf5"):
        infiles.append(os.path.join(input_dirpath, filename))
    os.chdir(workdir)
    return infiles


def write_to_outfile(involume, outvolume, indset, outfiles_partition, outdir_path, O):
    from dask_io.optimizer.utils.utils import numeric_to_3d_pos

    # open out file
    _3d_pos = numeric_to_3d_pos(outvolume.index, outfiles_partition, order='F')
    i, j, k = _3d_pos
    out_filename = f'{i}_{j}_{k}.hdf5'

    outfilepath = os.path.join(outdir_path, out_filename)
    if not os.path.isfile(outfilepath):
        f = h5py.File(outfilepath, 'w')
    else:
        f = h5py.File(outfilepath, 'r+')

    # if no datasets, create one
    print("KEYS", list(f.keys()))
    if not "/data" in f.keys():
        # print('[debug] No dataset, creating dataset')
        null_arr = np.zeros(O)
        outdset = f.create_dataset("/data", O, data=null_arr, dtype=np.float16)  # initialize an empty dataset
    else:
        # print('[debug] Dataset exists')
        outdset = f["/data"]

    # find subarray crossing both files
    subarr = get_overlap_subarray(involume, outvolume)
    lowcorner, upcorner = subarr

    # write subarray from infile to outfile and close
    slices = [(lowcorner[0], upcorner[0]), (lowcorner[1], upcorner[1]), (lowcorner[2], upcorner[2])]

    offset_in = involume.get_corners()[0]  # lower corner of input file
    offset_out = outvolume.get_corners()[0]

    slices_in_infile = [
        (lowcorner[0]-offset_in[0], upcorner[0]-offset_in[0]), 
        (lowcorner[1]-offset_in[1], upcorner[1]-offset_in[1]), 
        (lowcorner[2]-offset_in[2], upcorner[2]-offset_in[2])]
    
    slices_in_outfile = [
        (lowcorner[0]-offset_out[0], upcorner[0]-offset_out[0]), 
        (lowcorner[1]-offset_out[1], upcorner[1]-offset_out[1]), 
        (lowcorner[2]-offset_out[2], upcorner[2]-offset_out[2])]

    s = slices_in_infile
    s2 = slices_in_outfile

    # print(f"[debug] extracting {s[0][0]}:{s[0][1]}, {s[1][0]}:{s[1][1]}, {s[2][0]}:{s[2][1]} from input file")
    # print(f"[debug] inserting {s2[0][0]}:{s2[0][1]}, {s2[1][0]}:{s2[1][1]}, {s2[2][0]}:{s2[2][1]} into output file {out_filename}")
    data = indset[s[0][0]:s[0][1],s[1][0]:s[1][1],s[2][0]:s[2][1]]
    outdset[s2[0][0]:s2[0][1],s2[1][0]:s2[1][1],s2[2][0]:s2[2][1]] = data
    f.close()

    with h5py.File(os.path.join(outdir_path, out_filename), 'r') as f:
        stored = f['/data'][s2[0][0]:s2[0][1],s2[1][0]:s2[1][1],s2[2][0]:s2[2][1]]
        # print(f"dataset after store: {f['/data'].value}")
        if np.allclose(stored, data):
            print("[success] data successfully stored.")
        else:
            print("[error] in data storage")


def find_associated_volume(infilepath, infiles_volumes, infiles_partition):
    from dask_io.optimizer.utils.utils import _3d_to_numeric_pos

    filename = infilepath.split('/')[-1]
    pos = filename.split('_')
    pos[-1] = pos[-1].split('.')[0]
    pos = tuple(list(map(lambda s: int(s), pos)))
    numeric_pos = _3d_to_numeric_pos(pos, infiles_partition, order='F')
    return infiles_volumes[numeric_pos]


def get_overlap_subarray(hypercube1, hypercube2):
    """ Refactor of hypercubes_overlap to return the overlap subarray
    """
    from dask_io.optimizer.cases.resplit_utils import Volume

    if not isinstance(hypercube1, Volume) or \
        not isinstance(hypercube2, Volume):
        raise TypeError()

    lowercorner1, uppercorner1 = hypercube1.get_corners()
    lowercorner2, uppercorner2 = hypercube2.get_corners()
    nb_dims = len(uppercorner1)
    
    subarray_lowercorner = list()
    subarray_uppercorner = list()
    for i in range(nb_dims):
        subarray_lowercorner.append(max(lowercorner1[i], lowercorner2[i]))
        subarray_uppercorner.append(min(uppercorner1[i], uppercorner2[i]))

    
    print(f"Overlap subarray : {subarray_lowercorner[0]}:{subarray_uppercorner[0]}, {subarray_lowercorner[1]}:{subarray_uppercorner[1]}, {subarray_lowercorner[2]}:{subarray_uppercorner[2]}")
    return (subarray_lowercorner, subarray_uppercorner)


def clean_directory(dirpath):
    """ Remove intermediary files from split/rechunk.
    """
    workdir = os.getcwd()
    os.chdir(dirpath)
    for filename in glob.glob("[0-9]*_[0-9]*_[0-9]*.hdf5"):
        os.remove(filename)
    os.chdir(workdir)


def rechunk_plain_python(indir_path, outdir_path, B, O, I, R):
    """ Naive rechunk implementation in plain python
    """
    globals()['dask_io'] = __import__('dask_io')
    
    from dask_io.optimizer.cases.resplit_utils import get_blocks_shape, get_named_volumes, hypercubes_overlap
    from dask_io.optimizer.utils.get_arrays import get_dataset, clean_files

    infiles_partition = get_blocks_shape(R, I)
    infiles_volumes = get_named_volumes(infiles_partition, I)
    outfiles_partition = get_blocks_shape(R, O)
    outfiles_volumes = get_named_volumes(outfiles_partition, O)

    t = time.time()
    for infilepath in get_input_files(indir_path):
        data = get_dataset(infilepath, '/data')
        involume = find_associated_volume(infilepath, infiles_volumes, infiles_partition)
        for outvolume in outfiles_volumes.values():
            if hypercubes_overlap(involume, outvolume):
                write_to_outfile(involume, outvolume, data, outfiles_partition, outdir_path, O)
        clean_files()  # close opened files
    t = time.time() - t
    return t