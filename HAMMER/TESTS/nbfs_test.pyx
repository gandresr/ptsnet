import networkx as nx

def run():
    G = nx.Graph()

    G.add_edges_from([(1,2),(2,3),(3,4),(4,5),(5,6),(2,7),(7,8),(8,9)])

    source = 1
    centrality = nx.eigenvector_centrality(G)
    width = 5
    for u, v in nx.bfs_beam_edges(G, source, centrality.get, width):
        print((u, v))
