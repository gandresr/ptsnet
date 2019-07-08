from pmoc import Mesh, Simulation, MOC_PATH, NODE_INT, NODE_TYPES, JUNCTION_INT
from pmoc import clk, run_interior_step
from pmoc import save_simulation, open_simulation
from time import time
from multiprocessing import Process
import pickle
from pprint import pprint

import numpy as np
import pylab as plt
import dis

# # Test segmentation and partitioning
T = 10

inp_file = MOC_PATH + "example_models/UTnet_v2_April2016.inp"

# save_simulation(sim)
results_FLOW_close = []
results_HEAD_close = []
results_FLOW_far = []
results_HEAD_far = []
time_steps = [i*0.005 for i in range(1, 5)]

# for dt in time_steps:
#     setting = np.linspace(1, 0, int(1/dt))
#     mesh = Mesh(inp_file, dt = dt, default_wave_speed = 1200)

#     N_CLOSE = mesh.junctions_int[JUNCTION_INT['n1'], mesh.junction_ids['J-9']]
#     N_FAR = mesh.junctions_int[JUNCTION_INT['n1'], mesh.junction_ids['J-260']]
#     sim = Simulation(mesh, int(T/dt), check_model=False)
#     sim.define_curve('V1', 'valve', curve_file=MOC_PATH+'valves/v_curve.csv')
#     clk.tic()
#     for t in range(sim.time_steps-1):
#         if t < len(setting):
#             sim.set_valve_setting('V1', setting[t])
#         sim.run_step()
#     clk.toc()
#     print(T, dt)
#     t2 = np.linspace(0, T, int(T/dt))
#     results_FLOW_close.append((t2, sim.flow_results[:,N_CLOSE]))
#     results_HEAD_close.append((t2, sim.head_results[:,N_CLOSE]))
#     results_FLOW_far.append((t2, sim.flow_results[:,N_FAR]))
#     results_HEAD_far.append((t2, sim.head_results[:,N_FAR]))

# with open('results_close', 'wb') as f:
#     pickle.dump((results_FLOW_close, results_HEAD_close), f)
# with open('results_far', 'wb') as f:
#     pickle.dump((results_FLOW_far, results_HEAD_far), f)

with open('results_close', 'rb') as f:
    results_close = pickle.load(f)
with open('results_far', 'rb') as f:
    results_far = pickle.load(f)

results_FLOW_close = results_close[0]
results_HEAD_close = results_close[1]
results_FLOW_far = results_far[0]
results_HEAD_far = results_far[1]

for i in range(len(results_FLOW_close)):
    plt.plot(results_FLOW_close[i][0], results_FLOW_close[i][1], label="dt = %.4f" % time_steps[i])
    plt.title("Flow at node J-9")
plt.show()
for i in range(len(results_HEAD_close)):
    plt.plot(results_HEAD_close[i][0], results_HEAD_close[i][1], label="dt = %.4f" % time_steps[i])
    plt.title("Head at node J-9")
plt.show()
for i in range(len(results_FLOW_far)):
    plt.plot(results_FLOW_far[i][0], results_FLOW_far[i][1], label="dt = %.4f" % time_steps[i])
    plt.title("Flow at node J-260 (far)")
plt.show()
for i in range(len(results_HEAD_far)):
    plt.plot(results_HEAD_far[i][0], results_HEAD_far[i][1], label="dt = %.4f" % time_steps[i])
    plt.title("Head at node J-260 (far)")
plt.show()
