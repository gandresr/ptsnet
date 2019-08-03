import matplotlib.pyplot as plt

from phammer.simulation.sim import HammerSimulation

inpfile = '/home/watsup/Documents/Github/phammer/example_files/PHFC_SIM_17_4_13.inp'

sim = HammerSimulation(inpfile, {
    'time_step' : 0.01,
    'duration' : 1 # TODO change only data structures affected by the change
})

sim.set_wave_speeds(1200)
sim.set_segments()