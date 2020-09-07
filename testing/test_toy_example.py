import matplotlib.pyplot as plt
import numpy as np
# import pandas as pd

import os
import pickle
import matplotlib.pyplot as plt
import ntpath
from phammer.simulation.sim import HammerSimulation
from time import time
from phammer.utils.io import get_root_path

ROOT = get_root_path()
duration = 25; time_step = 0.01
inpfile = os.path.join(ROOT, os.pardir, 'example_files', 'Tnet1.inp')

sim = HammerSimulation(
    inpfile = inpfile,
    settings = {
        'time_step' : time_step,
        'duration' : duration,
        'skip_compatibility_check' : False,
        'warnings_on' : False,
        'show_progress' : True,
        'period' : 0
    },
    default_wave_speed = 800)

sim.add_curve('V_BUTTERFLY', 'valve',
    [1, 0.8, 0.6, 0.4, 0.2, 0],
    [0.067, 0.044, 0.024, 0.011, 0.004, 0.   ])

valves = sim.wn.valve_name_list
sim.assign_curve_to('V_BUTTERFLY', valves)

SR = int(1//sim.settings.time_step)
for valve in valves:
    sim.define_valve_settings(valve, np.linspace(0, 1, SR), np.linspace(1, 0, SR))

sim.initialize()

tt = np.linspace(0, duration, sim.settings.time_steps)

sim.worker.profiler.start('total_sim_time')
while not sim.is_over:
    sim.run_step()
sim.worker.profiler.stop('total_sim_time')

# plt.plot(tt, sim['pipe.end'].flowrate['4'])
# plt.plot(tt, sim['pipe.start'].flowrate['4'])
# plt.plot(tt, sim['pipe.start'].flowrate['3'])
# plt.show()
