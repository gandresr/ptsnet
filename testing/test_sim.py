from phammer.simulation.sim import Simulation
import matplotlib.pyplot as plt
from time import time

input_file = '/home/watsup/Documents/Github/hammer-net/example_files/LoopedNet.inp'

sim = Simulation(input_file,
    duration = 20, # [s]
    time_step = 0.1, # [s]
    default_wave_speed = 1200,
    full_results = True)

t = time()
sim.start()
print(time() - t)
t = time()
sim.run_sim()
print(time() - t)

plt.plot(sim.Q)
plt.show()