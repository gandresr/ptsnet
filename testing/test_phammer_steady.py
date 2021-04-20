import matplotlib.pyplot as plt
from ptsnet.simulation.ic import get_initial_conditions
from time import time

inpfile = '/home/watsup/Documents/Github/ptsnet/example_files/PHFC_SIM_17_4_13.inp'


ic = get_initial_conditions(inpfile)
id = ic['node'].lloc('Blk_198876')
ic['node'].head['Blk_198876']
ic['node'].head[id] = 2