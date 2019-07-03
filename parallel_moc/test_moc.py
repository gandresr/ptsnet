from pmoc import Mesh, Simulation, MOC_PATH, NODE_INT, NODE_TYPES
from pmoc import Clock, run_interior_step
from time import time
from multiprocessing import Process
from pprint import pprint

import numpy as np
import matplotlib.pyplot as plt
import dis

clk = Clock()

# Test segmentation and partitioning
T = 10000
dt = 0.01

setting = np.linspace(1, 0, int(2/dt))
setting2 = np.linspace(1, 0, int(5/dt))
mesh = Mesh(MOC_PATH + "example_models/LoopedNet_valve.inp", dt = dt, default_wave_speed = 1200)
sim = Simulation(mesh, int(T/dt))
sim.define_valve_setting('10', setting=setting2, default_setting=0)
sim.define_valve_setting('11', setting=setting, default_setting=0)
sim.define_curve('11', 'valve', curve_file=MOC_PATH+'valves/v_curve.csv')
sim.run_simulation()

# sim.plot_results()
t2 = np.linspace(0, T, int(T/dt))
plt.plot(t2, sim.flow_results[:,276])
plt.plot(t2, sim.flow_results[:,128])
plt.show()