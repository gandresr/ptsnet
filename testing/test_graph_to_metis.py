import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

from phammer.partitioning.network import export_graph
from phammer.partitioning.partitioning import define_partitions

G = nx.Graph()
G.add_edge('0', '1')
G.add_edge('1', '2')
G.add_edge('2', 'N')
G.add_edge('N', '3')
G.add_edge('3', '4')
G.add_edge('4', '5')
G.add_edge('N', '6')
G.add_edge('6', '7')
G.add_edge('7', '8')

graph_file = '/home/watsup/Documents/Github/phammer/testing/results/test_graph'
nodes = np.array(['0', '1', '2', '3', '4', '5', '6', '7', '8', 'N'])
labels = np.arange(len(nodes))
index = {nodes[i] : i+1 for i in labels}
export_graph(G, graph_file, index = index)
p, g = define_partitions(graph_file, 2)

# Nodes in processor 0
nodes[p == 0]
# Nodes in processor 1
nodes[p == 1]
# Ghost nodes
nodes[g]