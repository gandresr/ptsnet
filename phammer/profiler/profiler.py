import numpy as np
import pandas as pd
from time import time

class Job:
    def __init__(self, label):
        self.restart()
        self.end_time = None
        self.label = label
        self.time_stamps = []

    def restart(self):
        self.start_time = time()
        if len(self.time_stamps) % 2 != 0:
            self.time_stamps[-1] = self.start_time
        else:
            self.time_stamps.append(self.start_time)

    def stop(self):
        if len(self.time_stamps) % 2 != 0:
            self.end_time = time()
            self.time_stamps.append(self.end_time)

    @property
    def duration(self):
        if len(self.time_stamps) >= 2:
            return self.time_stamps[-1] - self.time_stamps[-2]
        else:
            return None

class Profiler:
    def __init__(self, rank = 0):
        self.jobs = {}
        self.rank = rank

    def start(self, label, rank = 0):
        if label in self.jobs:
            self.jobs[label].restart()
        else:
            self.jobs[label] = Job(label)

    def stop(self, label):
        self.jobs[label].stop()