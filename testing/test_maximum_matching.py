import networkx as nx

G = nx.Graph()
G.add_edges_from([(0,1), (0,4), (0,3), (1,2), (1,4), (2,3), (3,5), (3,4), (4,5), (4,6), (5,6), (5,7), (6,7)])
mm = nx.algorithms.matching.max_weight_matching(G)
G.remove_edges_from(mm)
mm = nx.algorithms.matching.max_weight_matching(G)
G.remove_edges_from(mm)
mm = nx.algorithms.matching.max_weight_matching(G)
print(mm)