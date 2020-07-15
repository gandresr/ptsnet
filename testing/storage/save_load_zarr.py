import numpy as np
import os
import zarr

from phammer.arrays import ZarrArray
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.rank
num_processors = comm.size

num_points = 300000
T = 100

num_elements = num_points // num_processors
zarr_shape = (num_points, T)
comm.barrier()
store = zarr.DirectoryStore(os.path.join(os.getcwd(), 'zarr_array', 'x.zarr'))
data = np.random.rand(num_elements, T)
if rank == 0:
    z = zarr.open(store, 'w', shape = zarr_shape, chunks = (1, zarr_shape[1],), dtype = float)
comm.barrier()
z = zarr.open(store, mode = 'r+')
chunk_size = data.shape[0]
final_index = comm.scan(chunk_size)
initial_index = final_index - chunk_size
print(rank, initial_index, final_index, chunk_size, z.shape, data.shape)
z[initial_index:final_index,:] = data
