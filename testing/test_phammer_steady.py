import matplotlib.pyplot as plt
from phammer.simulation.steady_sim import get_initial_conditions
from time import time

inpfile = '/home/watsup/Documents/Github/hammer-net/example_files/PHFC_SIM_17_4_13.inp'

t = time()
ic = get_initial_conditions(inpfile)
print(time() - t)

Ke = ic['nodes'].emitter_coefficient
d = ic['nodes'].demand
print(ic['pumps'].A)
print(ic['pumps'].B)
print(ic['pumps'].C)

plt.plot(ic['valves'].initial_setting, 'o')
plt.show()