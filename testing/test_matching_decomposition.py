import wntr
import networkx as nx
input_file = '../example_files/PHFC_SIM_17_4_13.inp'
wn = wntr.network.WaterNetworkModel(input_file)

def get_matchings(G):
    i = 0
    while (G.number_of_edges() > 0):
        mm = nx.algorithms.matching.maximal_matching(G)
        G.remove_edges_from(mm)
        i += 1
        print(i)

G = wn.get_graph().to_undirected()
get_matchings(G)
