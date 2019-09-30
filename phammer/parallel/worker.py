from collections import deque
from phammer.simulation.init import Initializator
from phammer.simulation.sim import HammerSettings

class Worker:
    def __init__(self, inpfile, rank, settings):
        self.send_queue = deque()
        self.recv_queue = deque()
        self.rank = rank
        self.init = None
        if self.rank == 0:
            self.init = Initializator(
                inpfile,
                self.settings.skip_compatibility_check,
                self.settings.warnings_on)

    def set_wave_speeds(self):
        self.init

    def distribute_ic(self):
        pass