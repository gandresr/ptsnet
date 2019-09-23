from collections import deque

class Worker:
    def __init__(self, id, p, sim = None):
        self.send_queue = deque()
        self.recv_queue = deque()
        self.id = id
        self.p = p
        if self.id == 0:
            if sim is None:
                raise ValueError('A simulation has to be defined for the master worker')
        self.sim = sim

    def distribute_ic(self):
        pass
