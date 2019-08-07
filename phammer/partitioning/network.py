import networkx as nx
import numpy as np

def get_dense_graph(G, weights = None):
    # If graph is a MultiGraph, then, the keys for weights should be given by the name of the
    # link, else, the keys of weights are given by

    def add_dense_edge(N, n1, n2, old_graph, new_graph):
        na = n1
        for i in range(N):
            nb = str(n1) + str(i) + str(n2)
            new_graph.add_edge(na, nb)
            na = nb

    D = nx.Graph()

    if weights is None:
        raise ValueError("'weights' not defined")

    for edge in G.edges:
        if type(G) == nx.MultiGraph:
            edge_name = edge[2]
        else:
            edge_name = edge

        add_dense_edge(weights[edge_name], edge[0], edge[1], G, D)

    return D

def export_graph(G, filename, index = None):
    if index is None:
        index = {node : i+1 for i, node in enumerate(G.nodes)}
    with open(filename + '.graph', 'w') as f:
        f.write("%d %d\n" % (len(G), len(G.edges)))
        for node in index:
            fline = ''
            for neighbor in G[node]:
                fline += "%d " % index[neighbor]
            f.write(fline.rstrip() + '\n')