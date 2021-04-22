import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import wntr
#import tsnet

from ptsnet.simulation.sim import PTSNETSimulation
from time import time

duration = 0.00048259999999999997*3; time_step = 1
inpfile = '/home/gr24269/Documents/Github/ptsnet/example_files/BWSN1.inp'

sim = PTSNETSimulation(
    inpfile,
    {
        'time_step' : time_step,
        'duration' : duration,
        'skip_compatibility_check' : False,
    },
    period = 219,
    default_wave_speed = 1200)

sim.add_curve('V_BUTTERFLY', 'valve',
    [1, 0.8, 0.6, 0.4, 0.2, 0],
    [1e3, 0.044, 0.024, 0.011, 0.004, 0.   ])

valves = sim.wn.valve_name_list
sim.assign_curve_to('V_BUTTERFLY', valves)

sim.initialize()

idx = np.cumsum(sim.ic['pipe'].segments+1).astype(int)

with open('pipe_processors_%d.csv' % sim.worker.rank, 'w') as f:
    for i, pipe in enumerate(sim.wn.pipe_name_list):
        print(i, len(sim.wn.pipe_name_list))
        if i > 0:
            proc_pipe = sim.worker.processors[idx[i-1]:idx[i]]
        else:
            proc_pipe = sim.worker.processors[:idx[i]]
        counts = np.bincount(proc_pipe)
        f.write( "%s,%d\n" % (sim.ic['pipe'].ival(i), np.argmax(counts)) )
