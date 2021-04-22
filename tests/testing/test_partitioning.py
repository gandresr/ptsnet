import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import wntr

from ptsnet.simulation.sim import PTSNETSimulation
from ptsnet.parallel.partitioning import even
from time import time

duration = 200; time_step = 1
inpfile = '/home/watsup/Documents/Github/ptsnet/example_files/LoopedNet_valve.inp'

sim = PTSNETSimulation(
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
    [1.4, 1, 0.55, 0.25, 0.1, 0])

sim.assign_curve_to('V_BUTTERFLY', sim.wn.valve_name_list)

sim.initialize()

t = time()
while sim.worker.t < 1000:
    sim.worker.run_step()
print(t, time())