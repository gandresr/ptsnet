from mpi4py import MPI
import networkx as nx
import numpy as np


comm = MPI.COMM_WORLD

G = nx.Graph()
G.add_edges_from([(0,3), (1,3), (2,3), (3,4), (4,5), (4,7), (4,6), (4,11), (8,11), (9,11), (10,11)])

neighbors = np.array(list(G.neighbors(comm.rank))).astype(int)
dist_comm = comm.Create_dist_graph_adjacent(
    sources = neighbors,
    destinations = neighbors,
    sourceweights = neighbors + 1,
    destweights = np.ones(len(neighbors), dtype = int)*comm.rank + 1)

data = np.zeros(len(neighbors))
data[:] = dist_comm.neighbor_alltoall(np.ones(len(neighbors))*comm.rank)
print(comm.rank, data)