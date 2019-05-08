from segmentation import EPANET_network as Net
from time import time
from pprint import pprint
t = time()
network = Net("models/LoopedNet.inp")
print(time() - t)
t = time()
network.define_wavespeeds(default_wavespeed = 1200)
print(time() - t)
t = time()
network.define_segments(0.2)
print(time() - t)
t = time()
network.define_mesh()
print(time() - t)
t = time()
network.write_mesh()
print(time() - t)
pprint(network.order)