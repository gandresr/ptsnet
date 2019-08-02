import matplotlib.pyplot as plt
from phammer.simulation.sim import HammerSimulation

inpfile = '/home/watsup/Documents/Github/hammer-net/example_files/PHFC_SIM_17_4_13.inp'

sim = HammerSimulation(inpfile, {
    'time_step' : 0.01
})

sim.set_wave_speeds(1200)
plt.plot(sim.ic['pipes'].flowrate)
plt.show()