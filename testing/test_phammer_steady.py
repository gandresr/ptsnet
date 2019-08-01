import matplotlib.pyplot as plt
from phammer.simulation.steady_sim import get_initial_conditions
from time import time

inpfile = '/home/watsup/Documents/Github/hammer-net/example_files/LoopedNet_leak.inp'

t = time()
conditions, wn = get_initial_conditions(inpfile)
print(time() - t)
Ke = conditions['nodes'].emitter_coefficient
d = conditions['nodes'].demand
h = conditions['nodes'].head

plt.plot(conditions['valves'].direction, 'o')
plt.show()
