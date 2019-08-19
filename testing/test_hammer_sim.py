import matplotlib.pyplot as plt
import numpy as np

from phammer.simulation.sim import HammerSimulation

from time import time

duration = 1; time_step = 0.01
inpfile = '/home/watsup/Downloads/Tnet2.inp'
# inpfile = '/home/watsup/Documents/Github/phammer/example_files/PHFC_SIM_17_4_13.inp'
# duration = 50; time_step = 0.01
inpfile = '/home/watsup/Documents/Github/phammer/example_files/LoopedNet_valve.inp'

sim = HammerSimulation(inpfile, {
    'time_step' : time_step,
    'duration' : duration,
    'skip_compatibility_check' : True,
})

sim.set_wave_speeds(1200)
sim.initialize()

t = time()
while not sim.is_over:
    sim.run_step()
print(time()-t)

tt = np.linspace(0, duration, sim.settings.time_steps)
plt.plot(tt, sim.pipe_results.inflow.T)
plt.title("Flowrate in pipes")
plt.xlabel("Time [s]")
plt.ylabel("Flowrate $[m^3/s]$")
plt.show()

plt.plot(tt, sim.node_results.head.T)
plt.title("Head in nodes")
plt.xlabel("Time [s]")
plt.ylabel("Head $[m]$")
plt.show()