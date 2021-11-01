# Imports

import matplotlib.pyplot as plt
import numpy as np
from ptsnet.simulation.sim import PTSNETSimulation
from ptsnet.utils.io import get_example_path

# Create a simulation
sim = PTSNETSimulation(
    inpfile = get_example_path('TNET3_HAMMER'),
    settings={
        'partitioning_method':'bisection',
        'time_step' : 0.00005,
        'duration' : 1,
        'show_progress' : True
    },
)
sim.run()