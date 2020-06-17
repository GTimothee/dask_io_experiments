import dask
import dask.array as da
import numpy as np

if __name__ == "__main__":
    arr = da.random.normal(size=(1000,1000,1000))
    arr = arr.astype(np.float16)
    arr.compute()
    print("success")