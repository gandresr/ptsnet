import numpy as np
from phammer.simulation.util import is_iterable
import heapq as hq
from time import time

# def exchange_point_data(send_data, send_queue, recv_queue):
#     next_rcv = 0
#     next_send = 0

#     data = np.zeros(9)

#     t = time()
#     while next_rcv < len(recv_queue[rank]) or next_send < len(send_queue[rank]):
#         if rank in recv_queue:
#             if next_rcv < len(recv_queue[rank]):
#                 next = recv_queue[rank][next_rcv]
#                 data[next] = comm.recv(source = next)
#                 next_rcv += 1
#         if rank in send_queue:
#             if next_send < len(send_queue[rank]):
#                 send_to = send_queue[rank][next_send]
#                 comm.send(rank, send_to)
#                 next_send += 1