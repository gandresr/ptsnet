#%%
import matplotlib.pyplot as plt
import numpy as np

from ptsnet.simulation.sim import PTSNETSimulation

from time import time

duration = 0.1; time_step = 0.01
inpfile = '/home/watsup/Documents/Github/ptsnet/example_files/PHFC_SIM_17_4_13.inp'

sim = PTSNETSimulation(inpfile, {
    'time_step' : time_step,
    'duration' : duration
})

sim.set_wave_speeds(1200)
sim.initialize()
#%%
%%timeit
sim.set_valve_setting(sim.wn.valve_name_list, 0.3)

# Conclusions
# Tested with and without caching, cached implementation
# results in a x2 speedup for large element_name arrays (N>1000)
# for N<1000 the cached implementation performs slightly better
# but with almost equal simulation times compared to the non-cached


#%%
