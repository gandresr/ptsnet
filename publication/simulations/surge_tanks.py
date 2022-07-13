## Pump shut-off

import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from time import time
from ptsnet.simulation.sim import PTSNETSimulation
from ptsnet.utils.io import get_example_path

matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'
matplotlib.rcParams['font.size'] = 32

inpfile = get_example_path('TNET3')
global_dt = 0.005
global_wave_speed = 1200
def run_sim(with_surge=True, surge_type='open'):
    sim = PTSNETSimulation(
        inpfile = inpfile,
        settings = {
            'duration': 20,
            'time_step': global_dt,
            'period' : 0,
            'default_wave_speed' : global_wave_speed})
    sim.define_pump_operation('PUMP-172', initial_setting=1, final_setting=0, start_time=0, end_time=1)
    if with_surge:
        if surge_type == 'Open':
            sim.add_surge_protection('JUNCTION-34', 'open', 0.1)
        else:
            sim.add_surge_protection('JUNCTION-34', 'closed', 0.1, 0.24, 0.2)
    sim.run()
    plt.plot(sim['time'], sim['node'].head['JUNCTION-34'], label=surge_type+' Surge Tank', linewidth=4)

fig = plt.figure(figsize=(14.5, 8)); fig.clf()
run_sim(with_surge=False, surge_type='No')
run_sim(surge_type='Open')
plt.xlabel('Time [s]', fontsize=30)
plt.ylabel('Head [m]', fontsize=30)
plt.xlim(0,20)
plt.legend(fontsize=26)
plt.tight_layout()
plt.savefig('open_surge_tank.pdf')

fig = plt.figure(figsize=(14.5, 8)); fig.clf()
run_sim(with_surge=False, surge_type='No')
run_sim(surge_type='Closed')
plt.xlabel('Time [s]', fontsize=30)
plt.ylabel('Head [m]', fontsize=30)
plt.xlim(0,20)
plt.legend(fontsize=26)
plt.tight_layout()
plt.savefig('closed_surge_tank.pdf')