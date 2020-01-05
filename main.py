from dask_io_experiments.experiment_1.experiment1 import experiment as experiment1

if __name__ == "__main__":
    experiment1(debug_mode=False,
        nb_repetitions=1,
        hardwares=["hdd"],
        cube_types=['very_small'],
        physical_chunked_options=[False],
        chunk_types=['slabs', 'blocks'],
        scheduler_options=[False],
        optimization_options=[True])