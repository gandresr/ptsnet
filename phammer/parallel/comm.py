from collections import deque

def initialize_worker(rank, N, ic, curves, esettings):
    send_obj(rank, N)
    send_obj(rank, ic)
    send_obj(rank, curves)
    send_obj(rank, esettings)