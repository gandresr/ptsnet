import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import wntr

import plotly.graph_objs as go

from phammer.simulation.sim import HammerSimulation
from time import time

duration = 200; time_step = 0.1
# inpfile = '/home/watsup/Documents/Github/phammer/example_files/PHFC_SIM_17_4_13.inp'
# inpfile = '/home/watsup/Documents/Github/phammer/example_files/benchmarks/B4.inp'
inpfile = '/home/watsup/Documents/Github/phammer/example_files/LoopedNet_pump.inp'
# inpfile = '/home/watsup/Documents/Github/phammer/example_files/Tnet2.inp'

sim = HammerSimulation(inpfile, {
    'time_step' : time_step,
    'duration' : duration,
    'skip_compatibility_check' : True,
})
sim.set_wave_speeds(1200)
sim.add_curve('V_BUTTERFLY', 'valve',
    [1, 0.8, 0.6, 0.4, 0.2, 0],
    [0.061, 0.044, 0.024, 0.011, 0.004, 0.   ])

valves = sim.wn.valve_name_list
sim.assign_curve_to('V_BUTTERFLY', valves)
# sim.define_valve_settings(valves, np.linspace(0, 1, 50), np.linspace(1, 1, 50))

# sim.assign_curve_to('V_BUTTERFLY', ['V1', 'V-END'])
# sim.define_valve_settings('V1', np.linspace(0, 1, 10), np.linspace(1, 1, 10))
# sim.define_valve_settings('V-END', np.linspace(0, 5, 100), np.linspace(1, 1, 100))
sim.initialize()

t = time()
while not sim.is_over:
    sim.run_step()
print(time() - t)

tt = np.linspace(0, duration, sim.settings.time_steps)
plt.plot(tt, sim.pipe_results.inflow.T)
# plt.legend(sim.pipe_results._index_keys)
plt.title("Flowrate in pipes")
plt.xlabel("Time [s]")
plt.ylabel("Flowrate $[m^3/s]$")
plt.show()

plt.plot(tt, sim.node_results.head.T)
# plt.legend(sim.node_results._index_keys)
plt.title("Head in nodes")
plt.xlabel("Time [s]")
plt.ylabel("Head $[m]$")
plt.show()

plt.plot(tt, sim.node_results.head.T - sim.ic['node'].elevation)
# plt.legend(sim.node_results._index_keys)
plt.title("Pressure in nodes")
plt.xlabel("Time [s]")
plt.ylabel("Pressure $[m]$")
plt.show()

plt.plot(tt, sim.node_results.leak_flow.T)
# plt.legend(sim.node_results._index_keys)
plt.title("Leak flow in nodes")
plt.xlabel("Time [s]")
plt.ylabel("Leak flow $[m^3/s]$")
plt.show()

plt.plot(tt, sim.node_results.demand_flow.T)
# plt.legend(sim.node_results._index_keys)
plt.title("Demand flow in nodes")
plt.xlabel("Time [s]")
plt.ylabel("Demand $[m^3/s]$")
plt.show()

head = pd.DataFrame(sim.node_results.head.T, columns = sim.node_results._index_keys)
in_flow = pd.DataFrame(sim.pipe_results.inflow.T, columns = sim.pipe_results._index_keys)
wntr.graphics.network.network_animation(sim.wn, node_attribute = head, link_attribute = in_flow)
plt.show()