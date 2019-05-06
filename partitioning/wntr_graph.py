import wntr
import networkx as nx

class Simulation:
    def __init__(self, inp_file):
        self.wn = wntr.network.WaterNetworkModel(inp_file)
        self.sim = wntr.sim.EpanetSimulator(self.wn)
        self.results = self.sim.run_sim()
        self.network = self.wn.get_graph()


def get_segmented_graph(network_file, dt, wavespeed_file=None, default_wavespeed=1200):
    # Start EPANET 
    # network_file without .inp
    sim = Simulation(network_file + '.inp')
    G = sim.wn.get_graph() # Get network undirected graph

    # Get a dict with the lengths of the pipes in the network
    # This dict will be used to determine the number of partitions
    #   in-place
    N = sim.wn.query_link_attribute('length')
    if not wavespeed_file:
        for i in range(len(G.edges()))
        a = [default_wavespeed]*len(N) # wavespeeds

    # Get the maximum time steps for each pipe
    for i, p in enumerate(N):
        N[p] /= a[i]

    # Maximum dt in the system to capture waves in all pipes
    max_dt = N[min(N, key=N.get)]

    # Desired dt
    t_step = min(dt, max_dt)

    for i, p in enumerate(N):
        N[p] /= t_step
        # The wavespeed is adjusted to compensate the truncation error
        e = int(N[p])-N[p] # truncation error
        a[i] = a[i]/(1 + e/N[p])
        N[p] = int(N[p])

    G_segmented = nx.Graph()
    for i, node in enumerate(G.nodes()):
        for neighbor in G[node].keys():
            for p in G[node][neighbor].keys():
                if G[node][neighbor][p]['type'] != 'Pipe':
                    N[p] = 1
                nnodes = N[p] # Number of new nodes between node and neighbor
                n1 = node
                for j in range(1, nnodes): # N segments require N-1 new nodes 
                    n2 = p + '.' + str(j)
                    G_segmented.add_edge(n1, n2)
                    n1 = n2
                G_segmented.add_edge(n1, neighbor)

    return G_segmented, a

def write_metis_file(network_file, dt):
    # Writes network description with weights = to the segments
    #   assigned to each pipe in order to solve the method of 
    #   characteristics. The file is written using the METIS
    #   sintax (also compatible with KaHIP)

    G, a = get_segmented_graph(network_file, dt)
    boundary_node = None
    for node in G.nodes():
        if G.degree(node) == 1:
            boundary_node = node
            break

    ordered_nodes = {}
    # Index starts in i=1 to be consistent with the file specifications of METIS
    for i, node in enumerate(nx.dfs_preorder_nodes(G, source=boundary_node)):
        ordered_nodes[node] = i+1
    
    # Write file with labels in order, with the new wavespeeds
    with open(network_file + '.order', 'w') as f:
        is_ghost = 
        left_id =
        right_id =
        f.write("%d\n" % node)
    
    with open(network_file + '.graph', 'w') as f:
        f.write("%d %d\n" % (len(G), len(G.edges())))
        for i, node in enumerate(ordered_nodes.keys()):
            fc = "" # file content
            for neighbor in G[node].keys():
                fc += "%d " % ordered_nodes[neighbor]
            fc += '\n'
            f.write(fc)