import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import wntr
import os

from phammer.simulation.sim import HammerSimulation
from time import time

duration = 5; time_step = 1e-3
inpfile = os.getcwd() + '/../example_files/Tnet3.inp'

sim = HammerSimulation(
    inpfile,
    {
        'time_step' : time_step,
        'duration' : duration,
        'skip_compatibility_check' : False,
        'warnings_on' : False
    },
    period = 0,
    default_wave_speed = 1000)

sim.add_curve('V_BUTTERFLY', 'valve',
    [1, 0.8, 0.6, 0.4, 0.2, 0],
    [1e3, 0.044, 0.024, 0.011, 0.004, 0.   ])

valves = sim.wn.valve_name_list
sim.assign_curve_to('V_BUTTERFLY', valves)

# sim.define_valve_settings(valves[0], np.linspace(0, 5, 10), np.linspace(0, 1, 10))
# sim.define_pump_settings('pump', np.linspace(0, 1, 50), np.linspace(1, 0, 50))

sim.initialize()
# print(sim.worker.points)
# print(sim.rank, sim.worker.send_queue, sim.worker.recv_queue)
# sim.run_step()

# print(sim.worker.processors, sim.worker.rank)
# ppoint = np.argmax(sim.worker.mem_pool_points.head[:,0]-sim.worker.mem_pool_points.head[:,1])
# ppoint_diff = np.max(sim.worker.mem_pool_points.head[:,0]-sim.worker.mem_pool_points.head[:,1])
# locp = np.where(sim.where.points['are_boundaries'] == ppoint)[0]
# locp -= locp % 2
# pipe_idx = int(locp / 2)
# print(sim.ic['pipe'].ival(pipe_idx))
# print(sim.worker.point_properties.Cm[2070])
# print(sim.worker.point_properties.Bm[2070])
# print(sim.worker.point_properties.Cp[ppoint])
# print(sim.worker.point_properties.Bp[ppoint])

print(sim.worker.num_points - sim.num_points/4, 'EXTRA POINTS')
ttt = time()
while not sim.is_over:
    t1 = time()
    sim.run_step()
    # print(t1-time(), 'step', sim.rank)
print('TOTAL TIME', sim.rank, time()-ttt)
print('global_barrier', np.mean(sim.worker.profiler.jobs['global_barrier'].durations), sim.rank)
print('local_barrier', np.mean(sim.worker.profiler.jobs['local_barrier'].durations), sim.rank)
# tm = tsnet.network.TransientModel(inpfile)
# tm.set_wavespeed(1200.) # m/s
# tm.set_time(duration)
# tm = tsnet.simulation.Initializer(tm, 0, 'DD')
# tm = tsnet.simulation.MOCSimulator(tm)

# n = tm.get_node('40')
# plt.plot(n.head)
# plt.show()
tt = np.linspace(0, duration, sim.settings.time_steps)
# plt.plot(tt, sim.worker.node_results.head.T, '-.')
# plt.plot(tm.simulation_timestamps, n.head)
# plt.show()
# plt.plot(tt, sim.worker.node_results.head['8'])
# plt.plot(tt, sim.worker.node_results.head['9'])
# plt.legend(['8','9'])
# plt.title("Head")
# plt.xlabel("Time [s]")
# plt.ylabel("Head [m]$")
# plt.show()

# tt = np.linspace(0, duration, sim.settings.time_steps)
# plt.plot(tt, sim.worker.pipe_start_results.flowrate.T)
# # plt.legend(sim.worker.pipe_start_results._index_keys)
# plt.title("Outflow in pipes")
# plt.xlabel("Time [s]")
# plt.ylabel("Flowrate $[m^3/s]$")
# plt.show()

plt.plot(tt, sim.worker.node_results.head.T)
# plt.legend(sim.worker.node_results._index_keys)
plt.title("Head in nodes")
plt.xlabel("Time [s]")
plt.ylabel("Head $[m]$")
plt.show()

# # # # # # plt.plot(tt, sim.node_results.head.T - sim.ic['node'].elevation)
# # # # # # plt.legend(sim.node_results._index_keys)
# # # # # # plt.title("Pressure in nodes")
# # # # # # plt.xlabel("Time [s]")
# # # # # # plt.ylabel("Pressure $[m]$")
# # # # # # plt.show()

# plt.plot(tt, sim.worker.node_results.leak_flow.T)
# plt.legend(sim.worker.node_results._index_keys)
# plt.title("Leak flow in nodes")
# plt.xlabel("Time [s]")
# plt.ylabel("Leak flow $[m^3/s]$")
# plt.show()

plt.plot(tt, sim.worker.node_results.demand_flow.T)
# plt.legend(sim.worker.node_results._index_keys)
plt.title("Demand flow in nodes")
plt.xlabel("Time [s]")
plt.ylabel("Demand $[m^3/s]$")
plt.show()

# # head = pd.DataFrame(sim.node_results.head.T, columns = sim.node_results._index_keys)
# # in_flow = pd.DataFrame(sim.pipe_results.inflow.T, columns = sim.pipe_results._index_keys)
# # wntr.graphics.network.network_animation(sim.wn, node_attribute = head, link_attribute = in_flow)
# # plt.show()