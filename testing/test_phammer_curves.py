import matplotlib.pyplot as plt
import numpy as np

from phammer.simulation.sim import HammerSimulation

from time import time

duration = 20; time_step = 0.01
inpfile = '/home/watsup/Documents/Github/phammer/example_files/LoopedNet.inp'
# duration = 0.1; time_step = 0.01
# inpfile = '/home/watsup/Documents/Github/phammer/example_files/PHFC_SIM_17_4_13.inp'

sim = HammerSimulation(inpfile, {
    'time_step' : time_step,
    'duration' : duration # TODO change only data structures affected by the change
})

sim.add_curve('V_BUTTERFLY', 'valve_curve',
    [1, 0.8, 0.6, 0.4, 0.2, 0],
    [1.4, 1, 0.55, 0.25, 0.1, 0])

sim.assign_curve_to('V_BUTTERFLY', '9')