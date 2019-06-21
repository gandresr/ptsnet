from pmoc import Mesh, Simulation, MOC_PATH
from pmoc import Clock, run_interior_step
from time import time
from multiprocessing import Process
from pprint import pprint

import matplotlib.pyplot as plt
import dis

clk = Clock()

# Test segmentation and partitioning
clk.tic()
mesh = Mesh(MOC_PATH + "example_models/LoopedNet.inp", dt = 0.1, default_wave_speed = 1200)
T = 20
sim = Simulation(mesh, T)
sim.define_curve('9', 'Valve', curve_file=MOC_PATH+'valves/v9_curve.csv')
sim.define_valve_setting('9', setting_file=MOC_PATH+'valves/v9_setting.csv')
clk.toc()
print(sim.discharge_coefficients[0])
print(sim.curves[0])