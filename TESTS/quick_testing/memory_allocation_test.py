import networkx as nx
import matplotlib.pyplot as plt

# Several memory allocation schemes are tested in order to minimize memory 
# coalescing

def create_grid(n, m):
    G = nx.DiGraph()
    for i in range(m):
        for j in range(n):
            if j < n-1:
                G.add_edge(i + j*m, i + (j + 1)*m)
            if i < m-1:
                G.add_edge(i + j*m, i + j*m + 1)
    return G

def segmented_grid(G, k):
    G_segmented = nx.Graph()
    for node in range(len(G.nodes())):
        for neighbor in G[node].keys():
            nnodes = k # Number of new nodes between node and neighbor
            n1 = node
            for j in range(1, nnodes): 
                n2 = str(node) + '.' + str(j) + '.' + str(neighbor)
                G_segmented.add_edge(n1, n2)
                n1 = n2
            G_segmented.add_edge(n1, neighbor)

    return G_segmented

G = create_grid(2, 4)
G_seg = segmented_grid(G, 2)
nx.draw(G_seg)
plt.show()

def write_metis_file(G, network_file):
    # Writes network description with weights = to the segments
    #   assigned to each pipe in order to solve the method of 
    #   characteristics. The file is written using the METIS
    #   sintax (also compatible with KaHIP)

    oorder = list(nx.dfs_preorder_nodes(G, source=0))
    print(oorder)
    order = {}
    for i, n in enumerate(oorder):
        order[n] = i + 1
    with open(network_file + '.graph', 'w') as f:
        f.write("%d %d\n" % (len(G), len(G.edges())))
        for i, node in enumerate(order.keys()):
            fc = "" # file content
            for neighbor in G[node].keys():
                fc += "%d " % order[neighbor]
            fc += '\n'
            f.write(fc)

write_metis_file(G_seg, 'grid')