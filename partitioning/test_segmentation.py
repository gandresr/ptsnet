from pmoc import MOC_network as Net
from time import time
from pprint import pprint
t = time()
network = Net("models/LoopedNet.inp")
print(time() - t)
t = time()
network.define_wavespeeds(default_wavespeed = 1200)
print(time() - t)
t = time()
network.define_segments(0.01)
print(time() - t)
t = time()
network.define_mesh()
print(time() - t)
t = time()
network.write_mesh()
print(time() - t)
t = time()
network.define_partitions(200)
print(time() - t)
network.define_initial_conditions()
print(network.hydraulic_model.pipe_resistance_coefficients) # Friction factor
