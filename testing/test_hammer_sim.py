import matplotlib.pyplot as plt
import numpy as np

from phammer.simulation.sim import HammerSimulation

from time import time

# duration = 1; time_step = 0.01
# inpfile = '/home/watsup/Downloads/Tnet2.inp'
# inpfile = '/home/watsup/Documents/Github/phammer/example_files/PHFC_SIM_17_4_13.inp'
duration = 200; time_step = 0.1
inpfile = '/home/watsup/Documents/Github/phammer/example_files/LoopedNet.inp'

sim = HammerSimulation(inpfile, {
    'time_step' : time_step,
    'duration' : duration,
    'skip_compatibility_check' : False,
})

sim.set_wave_speeds(1200)
sim.add_curve('V_BUTTERFLY', 'valve',
    [1, 0.8, 0.6, 0.4, 0.2, 0],
    [1.4, 1, 0.55, 0.25, 0.1, 0])

sim.assign_curve_to('V_BUTTERFLY', '9')
sim.define_valve_settings('9', np.linspace(0, 1, 10), np.linspace(1, 0.5, 10))
sim.initialize()

t = time()
while not sim.is_over:
    sim.run_step()
print(time()-t)

tt = np.linspace(0, duration, sim.settings.time_steps)
plt.plot(tt, sim.pipe_results.inflow.T)
plt.title("Flowrate in pipes")
plt.xlabel("Time [s]")
plt.ylabel("Flowrate $[m^3/s]$")
plt.show()

plt.plot(tt, sim.node_results.head.T)
plt.title("Head in nodes")
plt.xlabel("Time [s]")
plt.ylabel("Head $[m]$")
plt.show()