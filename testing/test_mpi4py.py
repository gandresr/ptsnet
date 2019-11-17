from mpi4py import MPI
import numpy as np
from time import time

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

send_queue = {
    0 : [],
    1 : [],
    2 : [],
    3 : [1],
    4 : [1],
    5 : [4],
    6 : [5],
    7 : [],
    8 : [5,7]
}

recv_queue = {
    0 : [],
    1 : [3,4],
    2 : [],
    3 : [],
    4 : [5],
    5 : [6,8],
    6 : [],
    7: [8],
    8 : [],
}

next_rcv = 0
next_send = 0

data = np.zeros(9)

t = time()
while next_rcv < len(recv_queue[rank]) or next_send < len(send_queue[rank]):
    if rank in recv_queue:
        if next_rcv < len(recv_queue[rank]):
            next = recv_queue[rank][next_rcv]
            data[next] = comm.recv(source = next)
            next_rcv += 1
    if rank in send_queue:
        if next_send < len(send_queue[rank]):
            send_to = send_queue[rank][next_send]
            comm.send(rank, send_to)
            next_send += 1

print(time()-t, rank, sum(data), recv_queue[rank], send_queue[rank])