import numpy as np
from time import time

p = 4

partitions = np.loadtxt('../partitioning/partitionings/p%d' % p, dtype=int)
separator = np.loadtxt('../partitioning/partitionings/s%d' % p, dtype=int)

nodes_order = {"n%d" % i : i for i in range(len(partitions))}
pp = partitions

new_order = [i for j in range(p) for i in np.where(partitions == j)[0]]

partitions = [partitions[i] for i in new_order]
separator = [separator[i] for i in new_order]
nodes_id = list(nodes_order.keys())
nodes_id = [nodes_id[i] for i in new_order]

nnorder = {}
norder = {}
for i, n in enumerate(nodes_id):
    norder[n] = i


for i, n in enumerate(norder):
    print(pp[nodes_order[n]], partitions[norder[n]], separator[norder[n]])

