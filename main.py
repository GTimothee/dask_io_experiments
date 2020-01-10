from dask_io_experiments.experiment_1.experiment1 import experiment as experiment1

def exp():
    experiment1(debug_mode=False,
        nb_repetitions=1,
        hardwares=["hdd"],
        cube_types=['big_brain'],
        physical_chunked_options=[False],
        chunk_types=['blocks'],
        scheduler_options=[False], # dont care with new function
        optimization_options=[False],
        nthreads_opti=[1],
        nthreads_non_opti=[1])


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
    from dask_io_experiments.seek_model import ClusteredCubicModel
    from dask_io_experiments.custom_setup import HDD_PATH
    from dask_io.cases.case_config import CaseConfig
    from dask_io.utils.utils import ONE_GIG
    import os
    import numpy as np

    buffer_size = 5.5 * ONE_GIG
    shape=(3850, 3025, 3500)
    chunks_shape=(770, 605, 700)
    chunk_dims = np.array(shape)/np.array(chunks_shape)
    chunk_dims = tuple(chunk_dims.reshape(1, -1)[0])
    print(chunk_dims)

    params = [shape, 
        chunks_shape, 
        chunk_dims, 
        np.dtype('float16'), 
        buffer_size]

    model = ClusteredCubicModel(*params)
    nb_seeks = model.get_nb_seeks()
    print(f'nb seeks: {nb_seeks}')


if __name__ == "__main__":
    seek_model()