import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

from time import time
from phammer.partitioning.network import get_denser_graph
from phammer.simulation.sim import HammerSimulation

inpfile = '/home/watsup/Documents/Github/phammer/example_files/PHFC_SIM_17_4_13.inp'
duration = 1; time_step = 0.01

t = time()
sim = HammerSimulation(inpfile, {
    'time_step' : time_step,
    'duration' : duration # TODO change only data structures affected by the change
})
print('HammerSimulation', time() - t)

t = time()
sim.set_wave_speeds(1200)
print('set_wave_speeds', time() - t)

t = time()
sim.initialize()
print('initialize', time() - t)

t = time()
D = get_denser_graph(sim.ic, sim.where)
print('get_denser_graph', time() - t)

nx.draw(D, pos=nx.circular_layout(D))
plt.show()