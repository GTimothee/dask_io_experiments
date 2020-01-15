from dask_io_experiments.experiment_1.experiment1 import experiment as experiment1
from dask_io_experiments.seek_model import ClusteredCubicModel

from dask_io.cases.case_config import CaseConfig
from dask_io.utils.utils import ONE_GIG

import numpy as np

def exp():
    all_buffer_sizes = [
        3 * ONE_GIG,
        6 * ONE_GIG, 
        9 * ONE_GIG, 
        12 * ONE_GIG, 
        15 * ONE_GIG
    ]

    experiment1(debug_mode=False,
        nb_repetitions=1,
        hardwares=["hdd"],
        cube_types=['small'],
        physical_chunked_options=[False],
        chunk_types=['blocks'],
        scheduler_options=[False], # dont care with new function
        optimization_options=[False, True],
        buffer_sizes=[
            3 * ONE_GIG,
            6 * ONE_GIG,],
        nthreads_opti=[1],
        nthreads_non_opti=[None])


def verify():
    """ Verify input array vs output split file (containing splits).
    TODO: add this feature as a script
    """
    from dask_io_experiments.experiment_1.experiment1 import test_goodness_split
    from dask_io_experiments.test_config import TestConfig

    options = [
        'hdd',
        'big_brain',
        'False',
        'blocks',
        'False', # dont care with new function
        'True',
        None, 
        None,
        (770, 605, 700),
    ]
    test_obj = TestConfig(options)
    test_obj.print_config()
    _ = test_goodness_split(getattr(test_obj, 'case'))


def seek_model():
    """
    TODO: add this feature as a script
    """
    
    cs_list = [
        (1000, 1000, 1000),
        (500, 500, 500),  # 10x11x10
        (100, 100, 100)
    ]

    mem_list = [
        3, 9, 15
    ]

    for mem_available in mem_list:
        for chunks_shape in cs_list:
            print(f'\n---------------------------')
            print(f'MEM: {mem_available}GB')
            print(f'CHUNKS SHAPE: {chunks_shape}')

            buffer_size = mem_available * ONE_GIG
            shape=(3000, 3000, 3000)
            
            chunk_dims = np.array(shape)/np.array(chunks_shape)
            chunk_dims = tuple(chunk_dims.reshape(1, -1)[0])
            print(chunk_dims)

            params = [shape, 
                chunks_shape, 
                chunk_dims, 
                np.dtype('float16'), 
                buffer_size]

            model = ClusteredCubicModel(*params)
            model.print_infos()
            nb_seeks = model.get_nb_seeks()
            print(f'nb seeks: {nb_seeks}')


def test_seek_model():
    shape = (20, 20, 20)
    chunks_shape = (5, 5, 5)
    chunk_dims = np.array(shape)/np.array(chunks_shape)
    chunk_dims = tuple(chunk_dims.reshape(1, -1)[0])
    dtype = np.dtype('int8')  # 1 byte (for simpler test case creation)
    _block_size = chunks_shape[0] * chunks_shape[1] * chunks_shape[2]
    _block_row_size = _block_size * chunk_dims[2]
    _block_slice_size = _block_row_size * chunk_dims[1]

    # in a F order file
    print(f'\nTest cases setup:')
    print(f'block size: {_block_size}')
    print(f'nb blocks per row: {chunk_dims[2]}')
    print(f'block_row_size: {_block_row_size}')
    print(f'nb_rows_per_slice: {chunk_dims[1]}')
    print(f'block_slice_size: {_block_slice_size}')

    params = {
        0: [shape, 
        chunks_shape, 
        chunk_dims, 
        dtype, 
        3 * _block_size],
        1: [shape, 
        chunks_shape, 
        chunk_dims, 
        dtype, 
        3 * _block_row_size],
        2: [shape, 
        chunks_shape, 
        chunk_dims, 
        dtype, 
        3 * _block_slice_size],
    }

    """  
        case 1:
        -------
        5x5: nb seeks for one buffer 
        2 buffers ber row: because 3*block_size and 4 blocks per row
        16 rows in the array: 4x4
        total = 5*5*2*16 + nb_splits

        case 2:
        -------
        5: nb seeks per buffer
        2: nb buffers per slice
        4: nb slices
        total: 5*2*4 + nb_splits

        case 3:
        -------
        1: nb seek per buffer
        2: nb buffers per image
        total = 2 + nb_splits
    """
    expected = [
        5*5*2*16 + 4*4*4,
        5*2*4 + 4*4*4,
        2 + 4*4*4
    ]

    for case in [0, 1, 2]:
        print("\n")
        model = ClusteredCubicModel(*params[case])
        print(model.get_strategy())
        print(f'\nModel output for nb seeks: {model.get_nb_seeks()}')
        print(f'Expected: {expected[case]} seeks.')


if __name__ == "__main__":
    exp()