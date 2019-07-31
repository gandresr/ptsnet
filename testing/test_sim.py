import matplotlib.pyplot as plt
import numpy as np

from phammer.simulation.sim import Simulation
from time import time

input_file = '/home/watsup/Documents/Github/hammer-net/example_files/LoopedNet.inp'

T = 20
dt = 0.01

sim = Simulation(input_file,
    duration = T, # [s]
    time_step = dt, # [s]
    default_wave_speed = 1200,
    full_results=True)

sim.start()
t = time()
sim.run_sim()
print(time() - t)

plt.plot(sim.H)
plt.show()