from pmoc import MOC_network as Net
from pmoc import MOC_simulation as Sim
from pmoc import Wall_clock as WC
from time import time
from pprint import pprint

clk = WC()

# Test segmentation and partitioning
clk.tic()
network = Net("models/single_pipe.inp")
network.define_wavespeeds(default_wavespeed = 1200)
network.define_segments(0.1)
network.define_mesh()
network.write_mesh()
network.define_partitions(2)
clk.toc()

# Test MOC
sim = Sim(network, 2)
clk.tic()
sim.define_initial_conditions()
clk.toc()
print(network.order)