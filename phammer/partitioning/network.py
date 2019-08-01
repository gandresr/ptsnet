import networkx as nx
from phammer.partitioning.constants import SUPPORTED_NETWORK_FORMATS

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