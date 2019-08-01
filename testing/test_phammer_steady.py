import matplotlib.pyplot as plt
from phammer.simulation.steady_sim import get_initial_conditions
from time import time

inpfile = '/home/watsup/Documents/Github/hammer-net/example_files/PHFC_SIM_17_4_13.inp'

t = time()
conditions = get_initial_conditions(inpfile)
print(time() - t)
Ke = conditions['nodes'].emitter_coefficient
d = conditions['nodes'].demand
print(conditions['pipes'].ffactor)

plt.plot(conditions['pipes'].ffactor, 'o')
plt.show()