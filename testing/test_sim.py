import matplotlib.pyplot as plt
import numpy as np

from phammer.simulation.sim import Simulation
from time import time

input_file = '/home/watsup/Documents/Github/hammer-net/example_files/PHFC_SIM_17_4_13.inp'

T = 1
dt = 0.01

sim = Simulation(input_file,
    duration = T, # [s]
    time_step = dt, # [s]
    default_wave_speed = 1200,
    full_results=False)

sim.mesh._run_steady_state()