from time import time
from mpi4py import MPI
from ptsnet.parallel.comm import exchange_point_data
import numpy as np

comm = MPI.COMM_WORLD
rank = comm.rank

data = np.arange(10) + rank
if rank == 0:
    send_queue = {1 : [7, 8, 1]}
    recv_queue = {}
elif rank == 1:
    send_queue = {2 : [2, 3]}
    recv_queue = {0 : [7, 8, 1]}
elif rank == 2:
    send_queue = {}
    recv_queue = {1 : [2, 3]}

print(rank, send_queue, recv_queue)
exchange_point_data(data, rank, comm, send_queue, recv_queue)
print(data)