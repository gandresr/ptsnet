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
T = 100
dt1 = 0.01
dt2 = 0.02
# mesh = Mesh(MOC_PATH + "example_models/LoopedNet.inp", dt = dt1, default_wave_speed = 1200)
# sim = Simulation(mesh, int(T/dt1))
# # sim.define_valve_setting('9', setting_file=MOC_PATH+'valves/v9_setting.csv', default_setting=0.1)
# clk.tic()
# sim.run_simulation()
# clk.toc()
# t1 = np.linspace(0, 60, int(T/dt1))
# x1 = sim.head_results[:,6]

mesh = Mesh(MOC_PATH + "example_models/LoopedNet.inp", dt = dt2, default_wave_speed = 1200)
print("Nodes: %d" % mesh.num_nodes)
sim = Simulation(mesh, int(T/dt2))
print("Time steps: %d" % sim.time_steps)
clk.tic()
sim.run_simulation()
clk.toc()
x2 = sim.head_results[:,139]
t2 = np.linspace(0, T, int(T/dt2))

plt.plot(t2, x2)
plt.axis([0, T, 0, 250])
plt.show()
plt.plot(t2[1:], 1./t2[1:])
plt.show()