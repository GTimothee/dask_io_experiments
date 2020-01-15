# folder 1

Contains the outputs of the following experiment:

```
experiment1(debug_mode=False,
        nb_repetitions=3,
        hardwares=["hdd"],
        cube_types=['big_brain'],
        physical_chunked_options=[False],
        chunk_types=['blocks'],
        scheduler_options=[False], # dont care with new function
        optimization_options=[False, True],
        buffer_sizes=[
            3 * ONE_GIG,
            9 * ONE_GIG,  
            15 * ONE_GIG
        ],
        nthreads_opti=[1],
        nthreads_non_opti=[None])
```
