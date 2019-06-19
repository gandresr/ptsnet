from pmoc import Mesh, Simulation, MOC_PATH
from pmoc import Clock, run_interior_step
from time import time
from multiprocessing import Process
from pprint import pprint

import matplotlib.pyplot as plt
import dis

clk = Clock()

# Test segmentation and partitioning
mesh = Mesh(MOC_PATH + "example_models/LoopedNet.inp", dt = 0.2, default_wave_speed = 1200)
sim = Simulation(mesh, 3)

print(sim.mesh.nodes_int)
print(sim.mesh.link_name_list)
print(sim.head_results[0])
print(sim.flow_results[0])
link = mesh.wn.get_link('2')
start = link.start_node_name
print(sim.steady_state_sim.node['head'][start])
