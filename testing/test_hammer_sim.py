import matplotlib.pyplot as plt
import numpy as np

from phammer.simulation.sim import HammerSimulation

from time import time

duration = 50; time_step = 0.1
inpfile = '/home/watsup/Documents/Github/phammer/example_files/LoopedNet.inp'
# duration = 0.1; time_step = 0.01
# inpfile = '/home/watsup/Documents/Github/phammer/example_files/PHFC_SIM_17_4_13.inp'

sim = HammerSimulation(inpfile, {
    'time_step' : time_step,
    'duration' : duration
})

sim.set_wave_speeds(1200)
sim.initialize()

t = time()
while not sim.is_over:
    sim.run_step()
print(time()-t)

plt.plot(sim.pipe_results.inflow['2'])
plt.plot(sim.pipe_results.outflow['2'])
plt.show()
plt.plot(sim.node_results.head['1'])
plt.show()
