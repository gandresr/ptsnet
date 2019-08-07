import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

from phammer.partitioning.network import export_graph
from phammer.partitioning.partitioning import define_partitions

G = nx.Graph()
G.add_edge('0', '1')
G.add_edge('1', '3')
G.add_edge('2', '3')
G.add_edge('3', '4')
G.add_edge('3', '5')
G.add_edge('5', '6')
G.add_edge('5', '7')
G.add_edge('7', '8')
G.add_edge('7', '10')
G.add_edge('9', '10')
G.add_edge('10', '12')
G.add_edge('10', '11')
G.add_edge('12', '13')
G.add_edge('12', '14')

graph_file = '/home/watsup/Documents/Github/phammer/testing/results/test_graph'
nodes = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14']
labels = np.arange(len(nodes))
index = {nodes[i] : i for i in labels}
export_graph(G, graph_file, index = index)
p, g = define_partitions(graph_file, 2)

# Nodes in processor 0
labels[p == 0]
# Nodes in processor 1
labels[p == 1]
# Ghost nodes
labels[g]