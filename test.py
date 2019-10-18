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

sim.add_curve('V_BUTTERFLY', 'valve',
    [1, 0.8, 0.6, 0.4, 0.2, 0],
    [1.4, 1, 0.55, 0.25, 0.1, 0])

sim.assign_curve_to('V_BUTTERFLY', '9')

sim.initialize()