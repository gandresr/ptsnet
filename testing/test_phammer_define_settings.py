import matplotlib.pyplot as plt
import numpy as np

from phammer.simulation.sim import HammerSimulation

from time import time

duration = 0.5; time_step = 0.01
inpfile = '/home/watsup/Documents/Github/phammer/example_files/PHFC_SIM_17_4_13.inp'

sim = HammerSimulation(inpfile, {
    'time_step' : time_step,
    'duration' : duration
})

sim.set_wave_speeds(1200)
valves = sim.wn.valve_name_list
x = np.linspace(0, 1, 100)
sim.define_valve_settings(valves[0], x, np.arange(len(x)))

sim.initialize()
sim.element_settings['valve'].activation_times
sim.element_settings['valve'].values
