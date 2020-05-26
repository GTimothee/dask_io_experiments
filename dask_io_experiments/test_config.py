import os
import traceback

from dask_io.optimizer.cases.case_config import Split, Merge
from dask_io.optimizer.utils.utils import ONE_GIG


class TestConfig():
    def __init__(self, params, paths):
        self.params = params
        self.create_split_case(params, paths)
        self.create_merge_case(params, paths)


    def create_split_case(self, params, paths):
        try: 
            if params["hardware"] == "hdd":
                self.hardware_path = paths["hdd_path"]
            else:
                self.hardware_path = paths["ssd_path"]
            self.cuboid_filepath = os.path.join(self.hardware_path, params["cuboid_name"] + ".hdf5")
            self.splitcase = Split(self.cuboid_filepath, params["chunk_shape"])
            self.splitcase.split_hdf5_multiple(self.hardware_path, nb_blocks=None)

        except Exception as e:
            print(traceback.format_exc())
            print("Something went wrong while creating case config.")
            exit(1)


    def create_merge_case(self, params, paths):
        try: 
            if params["hardware"] == "hdd":
                self.hardware_path = paths["hdd_path"]
            else:
                self.hardware_path = paths["ssd_path"]

            self.merge_filepath = os.path.join(self.hardware_path, "merged.hdf5")
            self.mergecase = Merge(self.merge_filepath)
            self.mergecase.merge_hdf5_multiple(self.hardware_path, data_key='/data', store=True)

        except Exception as e:
            print(traceback.format_exc())
            print("Something went wrong while creating case config.")
            exit(1)

    def print_config(self):
        print(f'\n-------------------')
        print(f'Test configuration')
        print(f'-------------------')
    
        print(f'\nTest configurations:')
        print(f'\tHardware: {self.params["hardware"]}')
        print(f'\tCuboid name: {self.params["cuboid_name"]}')
        print(f'\tCuboid shape: "{self.params["array_shape"]}"')
        print(f'\tChunk shape: "{self.params["chunk_shape"]}"')
        print(f'\tChunk type: "{self.params["chunk_type"]}"')

        print(f'\nDask configuration:')
        print(f'\tOptimization enabled: {self.params["optimized"]}')
        print(f'\tBuffer size: {self.params["buffer_size"]} bytes')
        print(f'\tNb threads: {self.params["nthreads"]}')       
        return