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
T = 400
dt1 = 0.01

mesh = Mesh(MOC_PATH + "example_models/LoopedNet_1.inp", dt = dt1, default_wave_speed = 1200)
sim = Simulation(mesh, int(T/dt1))
sim.define_valve_setting('9', setting_file=MOC_PATH+'valves/v9_setting.csv', default_setting=1)
sim.define_valve_setting('10', setting_file=MOC_PATH+'valves/v10_setting.csv', default_setting=1)
sim.define_curve('9', 'valve', curve_file=MOC_PATH+'valves/v_curve.csv')
sim.define_curve('10', 'valve', curve_file=MOC_PATH+'valves/v_curve.csv')
sim.run_simulation()

t2 = np.linspace(0, T, int(T/dt1))
plt.plot(t2, sim.head_results[:,0])
plt.show()