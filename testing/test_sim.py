from phammer.simulation.sim import Simulation
import matplotlib.pyplot as plt
from time import time

input_file = 'example_files/LoopedNet.inp'

sim = Simulation(input_file,
    duration = 20, # [s]
    time_step = 0.01, # [s]
    default_wave_speed = 1200,
    full_results = True)

sim.start()
t = time()
sim.run_sim()
print(time() - t)

