import numpy as np
import pandas as pd
from time import time

class Job:
    def __init__(self, label):
        self.restart()
        self.end_time = None
        self.label = label
        self.durations = []

    def restart(self):
        self.start_time = time()

    def stop(self):
        self.end_time = time()
        self.durations.append(self.end_time - self.start_time)

    @property
    def duration(self):
        if self.durations:
            return self.durations[-1]
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