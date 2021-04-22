import numpy as np
import pandas as pd

from time import time, clock
from ptsnet.arrays import Table
from ptsnet.simulation.constants import COEFF_TOL, STEP_JOBS, INIT_JOBS, COMM_JOBS

class Job:
    def __init__(self, label):
        self.time_stamps = []
        self.restart()
        self.end_time = None
        self.label = label

    def restart(self):
        self.start_time = clock()
        if len(self.time_stamps) % 2 != 0:
            self.time_stamps[-1] = self.start_time
        else:
            self.time_stamps.append(self.start_time)

    def stop(self):
        if len(self.time_stamps) % 2 != 0:
            self.end_time = clock()
            self.time_stamps.append(self.end_time)

    @property
    def duration(self):
        if len(self.time_stamps) >= 2:
            tstamps = np.array(self.time_stamps)
            return tstamps[1::2] - tstamps[::2]
        else:
            return None

class Profiler:

    def __init__(self, rank = 0, is_on = False, _super = None):
        self.jobs = {}
        self.rank = rank
        self.is_on = is_on
        self._super = _super
        if not _super is None:
            self.storer = self._super.storer
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
        if self.storer.exists('summary'):
            self.summary = self.storer.load_data('summary')
        else:
            raw_step_times = self.storer.load_data('raw_step_times')[:]
            ssize = len(STEP_JOBS)

            # hack for old format
            if raw_step_times.shape[0] % ssize != 0: ssize += 1
            num_processors = int(raw_step_times.shape[0] / ssize)

            raw_init_times = self.storer.load_data('raw_init_times')[:]
            isize = len(INIT_JOBS)

            if num_processors > 1:
                raw_comm_times = self.storer.load_data('raw_comm_times')[:]
                csize = len(COMM_JOBS)

            step_jobs = {job : np.float for job in STEP_JOBS}
            self.summary['step_jobs'] = Table(step_jobs, raw_step_times.shape[1])
            critical_step_processors = np.argmax(raw_step_times[::ssize], axis = 0)

            init_jobs = {job : np.float for job in INIT_JOBS}
            self.summary['init_jobs'] = Table(init_jobs, raw_init_times.shape[1])

            comm_jobs = {job : np.float for job in COMM_JOBS}
            if num_processors > 1:
                rsshape = raw_comm_times.shape[1]
            else:
                rsshape = 1
            self.summary['comm_jobs'] = Table(comm_jobs, rsshape)

            for i, job in enumerate(step_jobs):
                self.summary['step_jobs'].__dict__[job][:] = \
                    raw_step_times[i::ssize][critical_step_processors, np.arange(raw_step_times.shape[1], dtype = int)]
                self.summary[job] = sum(self.summary['step_jobs'].__dict__[job])
            for i, job in enumerate(init_jobs):
                self.summary['init_jobs'].__dict__[job][:] = \
                    np.max(raw_init_times[i::isize], axis = 0)
                self.summary[job] = sum(self.summary['init_jobs'].__dict__[job])
            for i, job in enumerate(comm_jobs):
                if num_processors > 1:
                    self.summary['comm_jobs'].__dict__[job][:] = \
                        np.max(raw_comm_times[i::csize], axis = 0)
                    self.summary[job] = sum(self.summary['comm_jobs'].__dict__[job])
                else:
                    self.summary['comm_jobs'].__dict__[job][:] = 0
                    self.summary[job] = 0

            self.storer.save_data('summary', self.summary, comm = 'main')
