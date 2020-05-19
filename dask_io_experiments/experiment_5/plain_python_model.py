import os, glob, h5py

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
    _3d_pos = numeric_to_3d_pos(outvolume.index, outfiles_partition, order='C')
    i, j, k = _3d_pos
    out_filename = f'{i}_{j}_{k}.hdf5'
    f = h5py.File(os.path.join(outdir_path, out_filename), 'w')

    # if no datasets, create one
    if not "/data" in f.keys():
        outdset = f.create_dataset("/data", O, dtype='f16')  # initialize an empty dataset
    else:
        outdset = f["/data"]

    # find subarray crossing both files
    subarr = get_overlap_subarray(involume, outvolume)

    # write subarray from infile to outfile and close
    lowcorner, upcorner = subarr
    slices = [(lowcorner[0], upcorner[0]), (lowcorner[1], upcorner[1]), (lowcorner[2], upcorner[2])]
    offset = involume.get_corners()[0]  # lower corner of input file
    slices_in_infile = [
        (lowcorner[0]-offset[0], upcorner[0]-offset[0]), 
        (lowcorner[1]-offset[1], upcorner[1]-offset[1]), 
        (lowcorner[2]-offset[2], upcorner[2]-offset[2])]
    s = slices_in_infile
    data = indset[s[0][0]:s[0][1],s[1][0]:s[1][1],s[2][0]:s[2][1]]
    outdset[slices[0][0]:slices[0][1],slices[1][0]:slices[1][1],slices[2][0]:slices[2][1]] = data
    f.close()


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

    return (subarray_lowercorner, subarray_uppercorner)

def rechunk_plain_python(indir_path, outdir_path, B, O, I, R):
    """ Naive rechunk implementation in plain python
    """
    globals()['dask_io'] = __import__('dask_io')
    
    from dask_io.optimizer.cases.resplit_utils import get_blocks_shape, get_named_volumes, hypercubes_overlap
    from dask_io.optimizer.utils.get_arrays import get_dataset

    infiles_partition = get_blocks_shape(R, I)
    infiles_volumes = get_named_volumes(infiles_partition, I)
    outfiles_partition = get_blocks_shape(R, O)
    outfiles_volumes = get_named_volumes(outfiles_partition, O)

    for infilepath in get_input_files(indir_path):
        data = get_dataset(infilepath, '/data')
        involume = find_associated_volume(infilepath, infiles_volumes, infiles_partition)
        for outvolume in outfiles_volumes.values():
            if hypercubes_overlap(involume, outvolume):
                write_to_outfile(involume, outvolume, data, outfiles_partition, outdir_path, O)