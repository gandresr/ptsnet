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
    'duration' : duration
})

sim.add_curve('V_BUTTERFLY', 'valve_curve',
    [1, 0.8, 0.6, 0.4, 0.2, 0],
    [1.4, 1, 0.55, 0.25, 0.1, 0])

sim.add_curve('V_SETTING', 'valve_setting',
    [0, 5, 20], [0, 0, 1])

sim.assign_curve_to('V_BUTTERFLY', '9')
sim.assign_curve_to('V_SETTING', '9')
sim.set_wave_speeds(1200)
sim.initialize()

t = time()
while not sim.is_over:
    sim.run_step()
    sim.set_setting('valve', '9', 0, 0.2)
print(time() - t)

plt.plot(sim.node_results.head['1'])
plt.show()