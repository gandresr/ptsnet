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
sim.define_valve_settings('10', np.linspace(0, 5, 50), np.linspace(1, 1, 50))
sim.initialize()

t = time()
while not sim.is_over:
    sim.run_step()
print('elapsed time [s]', time() - t, 's')

tt = np.linspace(0, duration, sim.settings.time_steps)
plt.plot(tt, sim.worker.pipe_start_results.flowrate.T)
plt.legend(sim.worker.pipe_start_results._index_keys)
plt.show()