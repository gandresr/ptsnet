import matplotlib.pyplot as plt
from phammer.simulation.ic import get_initial_conditions
from time import time

inpfile = '/home/watsup/Documents/Github/hammer-net/example_files/PHFC_SIM_17_4_13.inp'

ic = get_initial_conditions(inpfile)