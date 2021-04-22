import zarr
import numpy as np

from mpi4py import MPI
import random
from time import time

N = 10
labels = np.array(map(str, range(N)))
comm = MPI.COMM_WORLD
intra_comm = MPI.Intracomm(comm)
num_processors = comm.size
rank = comm.rank

labels = np.array(['node_%d' % i for i in range(10)])
processors = np.array([0,2,2,2,0,2,0,2,0,2])
my_labels = np.where(processors == rank)[0]

chunk_size = len(my_labels)

t = time()
if chunk_size > 0:
    final_index = intra_comm.scan(chunk_size)
    initial_index = final_index - chunk_size
    z = zarr.open('data/labels.zarr', mode='w', shape=N, dtype='<U20')
    z[initial_index:final_index] = labels[my_labels]
print(time()-t,rank)
print(rank, chunk_size, initial_index, final_index)