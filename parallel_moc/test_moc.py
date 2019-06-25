from pmoc import Mesh, Simulation, MOC_PATH, NODE_INT
from pmoc import Clock, run_interior_step
from time import time
from multiprocessing import Process
from pprint import pprint

import matplotlib.pyplot as plt
import dis

clk = Clock()

# Test segmentation and partitioning
clk.tic()
mesh = Mesh(MOC_PATH + "example_models/LoopedNet.inp", dt = 0.01, default_wave_speed = 1200)
sim = Simulation(mesh, 10000)
sim.define_valve_setting('9', setting_file=MOC_PATH+'valves/v9_setting.csv', default_setting=0.1)
clk.tic()
sim.run_simulation()
clk.toc()
plt.plot(sim.head_results)
plt.show()