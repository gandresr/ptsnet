#%%
from phammer.simulation.sim import Simulation
import matplotlib.pyplot as plt
from time import time

input_file = 'example_files/LoopedNet.inp'

def run_run(input_file):
    sim = Simulation(input_file,
        duration = 20, # [s]
        time_step = 0.01, # [s]
        default_wave_speed = 1200,
        full_results = False)
    sim.run_sim()

#%%
%%timeit
run_run(input_file)


#%%
