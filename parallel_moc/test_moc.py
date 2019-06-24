from pmoc import Mesh, Simulation, MOC_PATH
from pmoc import Clock, run_interior_step
from time import time
from multiprocessing import Process
from pprint import pprint

import matplotlib.pyplot as plt
import dis

clk = Clock()

# Test segmentation and partitioning
clk.tic()
mesh = Mesh(MOC_PATH + "example_models/LoopedNet.inp", dt = 0.1, default_wave_speed = 1200)
T = 50
print('Nodes: %d, Junctions: %d, Links: %d' % (mesh.num_nodes, mesh.num_junctions, mesh.num_links))
print('Time steps: %d' % T)
sim = Simulation(mesh, T)
sim.define_valve_setting('9', setting_file=MOC_PATH+'valves/v9_setting.csv', default_setting=1)
clk.toc()
sim.run_simulation()
print(mesh.junctions_int[2:4, mesh.junction_ids['6']])
print(mesh.nodes_int)
plt.plot(sim.flow_results[:,38])
plt.show()
plt.plot(sim.settings[0])
plt.show()