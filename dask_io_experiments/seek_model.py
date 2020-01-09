from dask_io.utils.array_utils import get_arr_shapes
import math

CLUSTERED_STRATEGIES = (
    "SINGLE_BLOCKS",
    "COMPLETE_ROWS",
    "COMPLETE_SLICES"
)

class ClusteredCubicModel():
    def __init__(self, arr, buffer_size):
        """ A dask array.
        """
        self.shape, self.chunks_shape, \
            self.chunk_dims, self.dtype = get_arr_shapes(arr, dtype=True)

        self.buffer_size = buffer_size
        self.nb_chunks_per_row = self.chunk_dims[0]
        self.nb_chunks = self.chunk_dims[0] **3
        self.nb_voxels = self.shape[0] **3
        self.arr_mem_size = self.nb_voxels * self.dtype.itemsize  # in bytes
        self.print_infos()

        self.nb_seeks_per_load = None 
        self.nb_loads = None
        self.memory_used = None
        return


    def print_infos(self):
        print(f'self.buffer_size : {self.buffer_size}')
        print(f'self.nb_chunks_per_row : {self.nb_chunks_per_row}')
        print(f'self.nb_chunks : {self.nb_chunks}')
        print(f'self.nb_voxels : {self.nb_voxels}')
        print(f'self.dtype.itemsize : {self.dtype.itemsize}')
        print(f'self.arr_mem_size : {self.arr_mem_size}')


    def get_strategy(self):
        self.strategy = None
        # TODO
        return self.strategy


    def get_memory_used(self):
        if not self.memory_used:
            if self.strategy == "SINGLE_BLOCKS":
                nb_bytes_per_chunk = self.arr_mem_size / self.nb_chunks
                nb_chunks_per_load = math.floor(self.buffer_size / nb_bytes_per_chunk)
                nb_bytes_per_obj = nb_bytes_per_chunk
                nb_obj_per_load = nb_chunks_per_load
            elif self.strategy == "COMPLETE_ROWS":
                nb_block_rows = self.nb_chunks_per_row **2
                nb_bytes_per_row = self.arr_mem_size / nb_block_rows
                nb_rows_per_load = math.floor(self.buffer_size / nb_bytes_per_row)
                nb_bytes_per_obj = nb_bytes_per_row
                nb_obj_per_load = nb_rows_per_load
            elif self.strategy == "COMPLETE_SLICES":
                nb_block_slices = self.nb_chunks_per_row
                nb_bytes_per_slice = self.arr_mem_size / nb_block_slices
                nb_slices_per_load = math.floor(self.buffer_size / nb_bytes_per_slice)
                nb_bytes_per_obj = nb_bytes_per_slice
                nb_obj_per_load = nb_slices_per_load
            else:
                raise ValueError('Strategy does not exist for clustered model.')
        
        self.memory_used = nb_bytes_per_obj * nb_obj_per_load
        return self.memory_used


    def get_nb_loads(self):
        if not self.nb_loads:
            if self.strategy == "SINGLE_BLOCKS":
                
            elif self.strategy == "COMPLETE_ROWS":

            elif self.strategy == "COMPLETE_SLICES":

            else:
                raise ValueError('Strategy does not exist for clustered model.')
        return self.nb_loads


    def get_nb_seeks_per_load():
        if not self.nb_seeks_per_load:
            if self.strategy == "SINGLE_BLOCKS":
                
            elif self.strategy == "COMPLETE_ROWS":

            elif self.strategy == "COMPLETE_SLICES":

            else:
                raise ValueError('Strategy does not exist for clustered model.')
        return self.nb_seeks_per_load


    def get_nb_seeks(self):
        """ Return the number of seeks introduced by splitting the input array.
        Main function to be called.
        """
        self.get_strategy()
        return self.nb_chunks + self.get_nb_loads() * self.get_nb_seeks_per_load()        



