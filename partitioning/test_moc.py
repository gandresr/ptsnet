from pmoc import MOC_network as Net
from pmoc import MOC_simulation as Sim
from pmoc import Wall_clock as WC
from time import time
from multiprocessing import Process
from pprint import pprint

clk = WC()

# Test segmentation and partitioning
network = Net("models/LoopedNet.inp")
network.define_wavespeeds(default_wavespeed = 1200)
network.define_segments(0.001)
network.define_mesh()
network.write_mesh()
# network.define_partitions(4)
network.define_partitions(4)

# Test MOC
T = 20
sim = Sim(network, T)

# for t in range(1, T):
#     sim.run_step(t, 0, len(network.mesh))

clk.tic()
num_threads = 3
N = [int(len(network.mesh)/num_threads) for i in range(num_threads-1)]
N.append(len(network.mesh) - sum(N))
for t in range(1,T):
    processes = []
    for i in range(num_threads):
        p = Process(target=sim.run_step, args=(t, i, N[i]))
        processes.append(p)
    # Start the processes
    for p in processes:
        p.start()
    # Ensure all processes have finished execution
    for p in processes:
        p.join()
clk.toc()

# print(sim.head_results)
# clk.tic()
# sim.define_valve_setting('9', 'valves/v9.csv')
# clk.toc()
