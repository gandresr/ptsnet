import wntr
import numpy as np
import matplotlib.pyplot as plt

from ptsnet.simulation.sim import PTSNETSimulation

sim = PTSNETSimulation()

s_pipes = list(map(str, sim['pipe.start'].labels.values()))
e_pipes = list(map(str, sim['pipe.end'].labels.values()))
nodes = list(map(str, sim['node'].labels.values()))

x1 = []
x2 = []
x3 = []

for pipe in s_pipes:
    x1.append(sim['pipe.start'].flowrate[pipe][0] - sim.ic['pipe'].flowrate[pipe])
    # print(sim['pipe.start'].flowrate[pipe][0], sim.ic['pipe'].flowrate[pipe])

for pipe in e_pipes:
    x2.append(sim['pipe.end'].flowrate[pipe][0] - sim.ic['pipe'].flowrate[pipe])
    # print(sim['pipe.end'].flowrate[pipe][0], sim.ic['pipe'].flowrate[pipe])

for node in nodes:
    x3.append(sim['node'].head[node][0] - sim.ic['node'].head[node])
    # print(node, sim['node'].head[node][0], sim.ic['node'].head[node])

x1 = np.array(x1)
x2 = np.array(x2)
x3 = np.array(x3)

plt.plot(x3)
plt.show()
assert np.all(x1 == 0)
assert np.all(x2 == 0)
assert np.all(x3 == 0)