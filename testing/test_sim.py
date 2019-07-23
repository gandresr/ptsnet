from phammer.simulation.sim import Simulation
import matplotlib.pyplot as plt
from time import time

input_file = '/home/watsup/Documents/Github/hammer-net/example_files/PHFC_SIM_17_4_13.inp'

sim = Simulation(input_file,
    duration = 2, # [s]
    time_step = 0.01, # [s]
    default_wave_speed = 1200,
    full_results = True)

t = time()
sim.start()
print(time() - t)
sim.run_sim()

plt.plot(sim.Q[:,0:10])
plt.show()