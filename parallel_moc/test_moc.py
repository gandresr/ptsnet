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
T = 40
dt1 = 0.01

clk.tic()
mesh = Mesh(MOC_PATH + "example_models/LoopedNet.inp", dt = dt1, default_wave_speed = 1200)
print("MESH CREATION")
clk.toc()
print(mesh.junctions_int)
print(mesh.junction_name_list)
print("Nodes: %d" % mesh.num_nodes)

clk.tic()
sim = Simulation(mesh, int(T/dt1))
print("INITIALIZATION")
clk.toc()
sim.define_valve_setting('9', setting_file=MOC_PATH+'valves/v9_setting.csv', default_setting=1)
print("Time steps: %d" % sim.time_steps)

clk.tic()
sim.run_simulation()
print("SIMULATION")
clk.toc()

t2 = np.linspace(0, T, int(T/dt1)) # 2
# plt.plot(t2, sim.head_results[:,0]) # 1
# plt.plot(t2, sim.head_results[:,128]) # 3
# plt.plot(t2, sim.head_results[:,276]) # 5
plt.plot(t2, sim.head_results[:,361]) # 4
# plt.plot(t2, sim.head_results[:,225]) # 6
plt.plot(t2, sim.head_results[:,275]) # 7
# plt.plot(t2, sim.head_results[:,400]) # 0
plt.show()
# plt.plot(t2, sim.flow_results[:,0]) # 1
# plt.plot(t2, sim.flow_results[:,128]) # 3
# plt.plot(t2, sim.flow_results[:,276]) # 5
# plt.plot(t2, sim.flow_results[:,361]) # 4
# plt.plot(t2, sim.flow_results[:,225]) # 6
# plt.plot(t2, sim.flow_results[:,275]) # 7
# plt.plot(t2, sim.flow_results[:,400]) # 0
# plt.show()