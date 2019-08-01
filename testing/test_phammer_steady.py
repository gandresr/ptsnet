import matplotlib.pyplot as plt
from phammer.simulation.steady_sim import get_initial_conditions, get_network_graph
from time import time

inpfile = '/home/watsup/Documents/Github/hammer-net/example_files/PHFC_SIM_17_4_13.inp'

G = get_network_graph(inpfile)