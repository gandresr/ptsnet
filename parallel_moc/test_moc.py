from pmoc import Mesh, Simulation, MOC_PATH
from pmoc import Clock, run_interior_step
from time import time
from multiprocessing import Process
from pprint import pprint

import matplotlib.pyplot as plt
import dis

clk = Clock()

# Test segmentation and partitioning
mesh = Mesh(MOC_PATH + "example_models/LoopedNet.inp", dt = 0.1, default_wave_speed = 1200)
print(mesh.junctions_int)
print(mesh.junctions_float)