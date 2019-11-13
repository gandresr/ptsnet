import numpy as np
from phammer.simulation.util import is_iterable
import heapq as hq
from time import time

class ReceiveQueue:
    def __init__(self):
        self.neighbors = {}
        self.data_points = set()

    def add_data(self, neighbor, data):
        if not neighbor in self.neighbors:
            self.neighbors[neighbor] = [] # heap
        if is_iterable(data):
            for d in data:
                if not d in self.data_points:
                    self.data_points.add(d)
                    hq.heappush(self.neighbors[neighbor], d)
        else:
            if not data in self.data_points:
                self.data_points.add(data)
                hq.heappush(self.neighbors[neighbor], data)

    def __getitem__(self, key):
        return self.neighbors[key] # can be popped in order using heapq

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