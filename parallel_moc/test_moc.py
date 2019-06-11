from pmoc import Mesh, Simulation
from pmoc import Clock
from time import time
from multiprocessing import Process
from pprint import pprint

clk = Clock()

# Test segmentation and partitioning
mesh = Mesh("example_models/LoopedNet.inp", dt = 0.1, default_wave_speed = 1200)
# mesh.define_partitions(4)

# Test MOC
T = 2
sim = Simulation(mesh, T)

# for t in range(1, T):
#     sim.run_step(t, 0, len(mesh.mesh_graph))

# print(sim.head_results)
# clk.tic()
# sim.define_valve_setting('9', 'valves/v9.csv')
# clk.toc()
