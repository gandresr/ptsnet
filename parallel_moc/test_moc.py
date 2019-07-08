from pmoc import Mesh, Simulation, MOC_PATH, NODE_INT, NODE_TYPES
from pmoc import clk, run_interior_step
from pmoc import save_model, open_model
from time import time
from multiprocessing import Process
from pprint import pprint

import numpy as np
import matplotlib.pyplot as plt
import dis

# Test segmentation and partitioning
T = 5000
dt = 0.05

inp_file = MOC_PATH + "example_models/LoopedNet_steady.inp"

mesh = Mesh(inp_file, dt = dt, default_wave_speed = 1200)
sim = Simulation(mesh, int(T/dt))
sim.define_curve('9', 'valve', curve_file=MOC_PATH+'valves/v_curve.csv')
# print(mesh.junctions_int)
# print(mesh.junction_name_list)

clk.tic()
for t in range(sim.time_steps-1):
    
    # if t < len(setting):
    #     sim.set_valve_setting('11', setting[t])
    sim.run_step()
clk.toc()

t2 = np.linspace(0, T, int(T/dt))
plt.plot(t2, sim.flow_results[:,0])
plt.show()