from pmoc import Mesh, Simulation, MOC_PATH
from pmoc import Clock
from time import time
from multiprocessing import Process
from pprint import pprint

clk = Clock()

# Test segmentation and partitioning
mesh = Mesh(MOC_PATH + "example_models/LoopedNet.inp", dt = 0.01, default_wave_speed = 1200)
mesh.define_partitions(4)

# print(mesh.node_name_list)
print(mesh.num_processors)
pprint(mesh.nodes)
# Test MOC
T = 2000
sim = Simulation(mesh, T)

# for t in range(1, T):
#     sim.run_step(t, 0, len(mesh.mesh_graph))

# print(sim.head_results)
# clk.tic()
# sim.define_valve_setting('9', 'valves/v9.csv')
# clk.toc()