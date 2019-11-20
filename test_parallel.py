import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import wntr

from phammer.simulation.sim import HammerSimulation
from time import time
from os import getcwd

duration = 200; time_step = 1
inpfile = getcwd() + '/example_files/LoopedNet_valve.inp'

sim = HammerSimulation(
    inpfile,
    {
        'time_step' : time_step,
        'duration' : duration,
        'skip_compatibility_check' : True,
        'warnings_on' : False,
    },
    default_wave_speed = 1200)

sim.add_curve('V_BUTTERFLY', 'valve',
    [1, 0.8, 0.6, 0.4, 0.2, 0],
    [0.0614, 0.044, 0.024, 0.011, 0.004, 0.   ])

valves = sim.wn.valve_name_list
sim.assign_curve_to('V_BUTTERFLY', valves)
sim.settings.num_processors = 4
sim.initialize()
print(sim.worker.rank, 's', sim.worker.send_queue.keys, sim.worker.send_queue.values)
print(sim.worker.rank, 'r', sim.worker.recv_queue.keys, sim.worker.recv_queue.values)
sim.run_step()