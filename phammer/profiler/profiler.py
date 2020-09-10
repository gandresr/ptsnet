import numpy as np
import pandas as pd

from time import time
from phammer.arrays import Table
from phammer.simulation.constants import COEFF_TOL, STEP_JOBS, INIT_JOBS, COMM_JOBS

class Job:
    def __init__(self, label):
        self.time_stamps = []
        self.restart()
        self.end_time = None
        self.label = label

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
            tstamps = np.array(self.time_stamps)
            return tstamps[1::2] - tstamps[::2]
        else:
            return None

class Profiler:

    def __init__(self, rank = 0, is_on = False, storer = None):
        self.jobs = {}
        self.rank = rank
        self.is_on = is_on
        self.storer = storer
        self.summary = {}

    def start(self, label, rank = 0):
        if not self.is_on: return
        if label in self.jobs:
            self.jobs[label].restart()
        else:
            self.jobs[label] = Job(label)

    def stop(self, label):
        if not self.is_on: return
        self.jobs[label].stop()

    def summarize_step_times(self):
        raw_step_times = self.storer.load_data('raw_step_times')
        num_processors = int(raw_step_times.shape[0] / len(STEP_JOBS))
        steps = raw_step_times.shape[1]

        raw_init_times = self.storer.load_data('raw_init_times')

        if num_processors > 1:
            raw_comm_times = self.storer.load_data('raw_comm_times')


        step_jobs = {job : np.float for job in STEP_JOBS}
        self.summary['step_jobs'] = Table(
                    step_jobs,
                    raw_step_times.shape[1])

        critical_step_processors = np.argmax(raw_step_times[::num_processors], axis = 0)

        init_jobs = {job : np.float for job in INIT_JOBS}
        self.summary['init_jobs'] = Table(
                    init_jobs,
                    raw_init_times.shape[1])

        if num_processors > 1:
            comm_jobs = {job : np.float for job in COMM_JOBS}
            self.summary['comm_jobs'] = Table(
                        comm_jobs,
                        raw_comm_times.shape[1])

        for i, job in enumerate(step_jobs):
            self.summary['step_jobs'].__dict__[job][:] = \
                raw_step_times[i::num_processors][critical_step_processors, np.arange(steps, dtype = int)]
            self.summary[job] = sum(self.summary['step_jobs'].__dict__[job])
        for i, job in enumerate(init_jobs):
            self.summary['init_jobs'].__dict__[job][:] = \
                np.max(raw_init_times[i::num_processors], axis = 0)
            self.summary[job] = sum(self.summary['init_jobs'].__dict__[job])
        if num_processors > 1:
            for i, job in enumerate(comm_jobs):
                self.summary['comm_jobs'].__dict__[job][:] = \
                    np.max(raw_comm_times[i::num_processors], axis = 0)
                self.summary[job] = sum(self.summary['comm_jobs'].__dict__[job])