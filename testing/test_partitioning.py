import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import wntr

from phammer.simulation.sim import HammerSimulation
from phammer.parallel.partitioning import get_points, even
from time import time

duration = 200; time_step = 1
inpfile = '/home/watsup/Documents/Github/phammer/example_files/LoopedNet.inp'

sim = HammerSimulation(
    inpfile,
    {
        'time_step' : time_step,
        'duration' : duration,
        'skip_compatibility_check' : True,
    },
    default_wave_speed = 1200)

N = sim.num_points
k = 4
rank = 0
processors = even(N, k) # This line depends on the partitioning function
p = get_points(processors, N, k, sim.where, 0)
print(p)

