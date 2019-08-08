import matplotlib.pyplot as plt
import numpy as np

from phammer.simulation.sim import HammerSimulation

from time import time

duration = 20; time_step = 0.01
inpfile = '/home/watsup/Documents/Github/phammer/example_files/LoopedNet_tank.inp'
# duration = 0.1; time_step = 0.01
# inpfile = '/home/watsup/Documents/Github/phammer/example_files/PHFC_SIM_17_4_13.inp'

t = time()
sim = HammerSimulation(inpfile, {
    'time_step' : time_step,
    'duration' : duration # TODO change only data structures affected by the change
})

sim.set_wave_speeds(1200)
sim.initialize()

t = time()
for i in range(1,sim.settings.time_steps):
    sim.run_step()
print(time()-t)

plt.plot(sim.pipe_results.inflow[2])
plt.plot(sim.pipe_results.outflow[2])
plt.show()
