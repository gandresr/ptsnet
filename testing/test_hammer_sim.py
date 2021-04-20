import numpy as np
import wntr
import os
import pickle
import matplotlib.pyplot as plt
import ntpath

from ptsnet.simulation.sim import PTSNETSimulation
from time import time
from ptsnet.utils.io import get_root_path

ROOT = get_root_path()
duration = 10; time_step = 1
inpfile = os.path.join(ROOT, os.pardir, 'example_files', 'PHFC_ws.inp')

sim = PTSNETSimulation(
    inpfile,
    {
        'time_step' : time_step,
        'duration' : duration,
        'skip_compatibility_check' : False,
        'warnings_on' : False,
        'show_progress' : True,
    },
    period = 0,
    default_wave_speed = 100)

sim.add_curve('V_BUTTERFLY', 'valve',
    [1, 0.8, 0.6, 0.4, 0.2, 0],
    [0.067, 0.044, 0.024, 0.011, 0.004, 0.   ])

valves = sim.wn.valve_name_list
pumps = sim.wn.pump_name_list
sim.assign_curve_to('V_BUTTERFLY', valves)

SR = int(1//sim.settings.time_step)
for valve in valves:
    sim.define_valve_settings(valve, np.linspace(0, 1, SR), np.linspace(1, 0, SR))

for pump in pumps:
    sim.define_pump_settings(pump, np.linspace(0, 1, SR), np.linspace(1, 0, SR))

sim.initialize()

sim.worker.profiler.start('total_sim_time')
while not sim.is_over:
    sim.run_step()
sim.worker.profiler.stop('total_sim_time')