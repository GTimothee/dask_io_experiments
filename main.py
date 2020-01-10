from dask_io_experiments.experiment_1.experiment1 import experiment as experiment1
from dask_io_experiments.experiment_1.experiment1 import test_goodness_split
from dask_io_experiments.test_config import TestConfig


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
    """ For debug purposes.
    """
    options = [
        'hdd',
        'big_brain',
        'False',
        'blocks',
        'False', # dont care with new function
        'True',
        (770, 605, 700)
    ]
    test_obj = TestConfig(options)
    test_obj.print_config()
    test_goodness_split(test_obj.case)

if __name__ == "__main__":
    exp()