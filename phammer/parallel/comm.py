import numpy as np
from phammer.simulation.util import is_iterable
import heapq as hq

def exchange_point_data(data, rank, comm, send_queue, recv_queue):
    next_rcv = 0
    next_send = 0
    rcv_keys = sorted(recv_queue.keys())
    send_keys = sorted(send_queue.keys())

    while next_rcv < len(rcv_keys) or next_send < len(send_keys):
        if next_rcv < len(rcv_keys):
            next = rcv_keys[next_rcv]
            print('before', rank, data)
            data[recv_queue[next]] = comm.recv(source = next)
            print('after', rank, data)
            next_rcv += 1
        if next_send < len(send_keys):
            send_to = send_keys[next_send]
            comm.send(data[send_queue[send_to]], send_to)
            next_send += 1