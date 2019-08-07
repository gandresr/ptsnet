from mpi4py import MPI
from time import time
import numpy as np

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

if rank == 0:
    N = 1000
    data = np.arange(N)
    comm.send(1, dest=1, tag=1)
    comm.send(data, dest=1, tag=11)
elif rank == 1:
    data = comm.recv(source=0, tag=1)
    t = time()
    data = comm.recv(source=0, tag=11)
    print(time() - t)
    print(data, rank)