from dask_io.utils.array_utils import get_arr_shapes
import math

CLUSTERED_STRATEGIES = (
    "SINGLE_BLOCKS",
    "COMPLETE_ROWS",
    "COMPLETE_SLICES"
)

class ClusteredCubicModel():
    def __init__(self, arr, buffer_size):
        """ 
        Arguments:
        ---------
            arr: A dask array.
            buffer_size: buffer size for clustered strategy.

        Usage:
        ------
            >> model = ClusteredCubicModel(arr, buffer_size)
            >> model.get_nb_seeks()
        """
        self.shape, self.chunks_shape, \
            self.chunk_dims, self.dtype = get_arr_shapes(arr, dtype=True)

        self.buffer_size = buffer_size
        self.nb_chunks_per_row = self.chunk_dims[0]
        self.nb_chunks = self.chunk_dims[0] **3
        self.nb_voxels = self.shape[0] **3
        self.arr_mem_size = self.nb_voxels * self.dtype.itemsize  # in bytes
        self.strategy = None
        self.strategy = self.get_strategy()

        self.print_infos()

        self.nb_seeks_per_load = None 
        self.nb_loads = None
        self.memory_used = None
        self.nb_seeks = None
        return


    def print_infos(self):
        print(f'buffer_size : {self.buffer_size}')
        print(f'nb_chunks_per_row : {self.nb_chunks_per_row}')
        print(f'nb_chunks : {self.nb_chunks}')
        print(f'nb_voxels : {self.nb_voxels}')
        print(f'dtype.itemsize : {self.dtype.itemsize}')
        print(f'arr_mem_size : {self.arr_mem_size}')
        print(f'strategy : {self.strategy}')


    def get_strategy(self):
        if not self.strategy:
            # find nb bytes per obj (block/row/slice)
            self.nb_bytes_per_chunk = self.arr_mem_size / self.nb_chunks

            nb_block_rows = self.nb_chunks_per_row **2
            self.nb_bytes_per_row = self.arr_mem_size / nb_block_rows

            nb_block_slices = self.nb_chunks_per_row
            self.nb_bytes_per_slice = self.arr_mem_size / nb_block_slices

            # find strategy
            if self.buffer_size < self.nb_bytes_per_row:
                self.strategy = CLUSTERED_STRATEGIES.SINGLE_BLOCKS
            elif self.buffer_size < self.nb_bytes_per_slice:
                self.strategy = CLUSTERED_STRATEGIES.COMPLETE_ROWS
            else:
                self.strategy = self.CLUSTERED_STRATEGIES.COMPLETE_SLICES    
        return self.strategy


    def get_memory_used(self):
        if not self.memory_used:
            if self.strategy == "SINGLE_BLOCKS":
                nb_chunks_per_load = math.floor(self.buffer_size / self.nb_bytes_per_chunk)
                nb_bytes_per_obj = nb_bytes_per_chunk
                nb_obj_per_load = nb_chunks_per_load
            elif self.strategy == "COMPLETE_ROWS":
                nb_rows_per_load = math.floor(self.buffer_size / self.nb_bytes_per_row)
                nb_bytes_per_obj = nb_bytes_per_row
                nb_obj_per_load = nb_rows_per_load
            elif self.strategy == "COMPLETE_SLICES":
                nb_slices_per_load = math.floor(self.buffer_size / self.nb_bytes_per_slice)
                nb_bytes_per_obj = nb_bytes_per_slice
                nb_obj_per_load = nb_slices_per_load
            else:
                raise ValueError('Strategy does not exist for clustered model.')
        
        self.memory_used = nb_bytes_per_obj * nb_obj_per_load
        return self.memory_used


    def get_nb_loads(self):
        if not self.nb_loads:
            if self.strategy == "SINGLE_BLOCKS":
                nb_rows = self.nb_chunks_per_row **2  # (=nb rows per slice * nb_slices)
                block_row_size = self.arr_mem_size / nb_rows
                self.nb_loads = math.ceil(block_row_size / self.memory_used) * nb_rows
            elif self.strategy == "COMPLETE_ROWS":
                nb_slices = self.nb_chunks_per_row
                slice_size = self.arr_mem_size / nb_slices
                self.nb_loads = math.ceil(slice_size / self.memory_used) * nb_slices
            elif self.strategy == "COMPLETE_SLICES":
                self.nb_loads = math.ceil(self.arr_mem_size / self.memory_used)
            else:
                raise ValueError('Strategy does not exist for clustered model.')
        return self.nb_loads


    def get_nb_seeks_per_load():
        if not self.nb_seeks_per_load:
            nbvoxels_per_blocklength = self.chunks_shape[0]
            if self.strategy == "SINGLE_BLOCKS":
                self.nb_seeks_per_load = nbvoxels_per_blocklength **2
            elif self.strategy == "COMPLETE_ROWS":
                self.nb_seeks_per_load = nbvoxels_per_blocklength
            elif self.strategy == "COMPLETE_SLICES":
                self.nb_seeks_per_load = 1
            else:
                raise ValueError('Strategy does not exist for clustered model.')
        return self.nb_seeks_per_load


    def get_nb_seeks(self):
        """ Return the number of seeks introduced by splitting the input array.
        Main function to be called.
        """
        if not self.nb_seeks:
            self.nb_seeks = self.nb_chunks + \
                self.get_nb_loads() * self.get_nb_seeks_per_load() 
        return self.nb_seeks        



