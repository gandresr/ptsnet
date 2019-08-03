import matplotlib.pyplot as plt
import numpy as np
from phammer.simulation.sim import HammerSimulation
from phammer.simulation.util import imerge
from phammer.epanet.util import EN
from time import time

# inpfile = '/home/watsup/Documents/Github/phammer/example_files/LoopedNet.inp'
inpfile = '/home/watsup/Documents/Github/phammer/example_files/PHFC_SIM_17_4_13.inp'

t = time()
sim = HammerSimulation(inpfile, {
    'time_step' : 0.01,
    'duration' : 1 # TODO change only data structures affected by the change
})
sim.set_wave_speeds(1200)
sim.set_segments()
print(time() - t)
t = time()
sim._create_selectors()
print(time() - t)