from pmoc import MOC_network as Net
from pmoc import MOC_simulation as Sim
from pmoc import Wall_clock as WC
from time import time
from pprint import pprint

clk = WC()

# Test segmentation and partitioning
clk.tic()
network = Net("models/LoopedNet.inp")
network.define_wavespeeds(default_wavespeed = 1200)
network.define_segments(0.1)
network.define_mesh()
network.write_mesh()
network.define_partitions(2)
clk.toc()

# Test MOC
T = 10
sim = Sim(network, T)
clk.tic()
sim.define_initial_conditions()
clk.toc()
sim.define_properties()

print(network.valves_order)
sim.define_valve_setting('9', 'valves/v9.csv')
print(sim.valve_settings)
