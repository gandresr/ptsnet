import wntr
import networkx as nx 
import matplotlib.pyplot as plt

class EPANET_network:
    def __init__(self, input_file):
        self.fname = input_file[:input_file.find('.inp')]
        self.wn = wntr.network.WaterNetworkModel(input_file)
        self.network = self.wn.get_graph()
        self.segmented_network = None
        self.a = {} # Wavespeed values
        # Segments are only defined for pipes
        self.segments = self.wn.query_link_attribute('length')
        self.order = {} # Ordered nodes
        # Used when sim is run
        self.sim = None
        self.results = None

    def run_sim(self):
        self.sim = wntr.sim.EpanetSimulator(self.wn)
        self.sim.run_sim()

    def define_wavespeeds(self, default_wavespeed = 1200, wavespeed_file = None):
        if wavespeed_file:
            f = open(wavespeed_file).read().split('\n')
            for p, a_p in map(lambda x : x.split(','), f):
                self.a[p] = float(a_p)
        else:
            for p, data in self.wn.links():
                if data.link_type != 'Pipe':
                    self.a[p] = 0
                else:
                    self.a[p] = default_wavespeed

    def define_segments(self, dt):
        # Get the maximum time steps for each pipe
        for p in self.segments:
            self.segments[p] /= self.a[p]
        
        # Maximum dt in the system to capture waves in all pipes
        max_dt = self.segments[min(self.segments, key=self.segments.get)]

        # Desired dt < max_dt ?
        t_step = min(dt, max_dt)

        # The number of segments are defined
        for p in self.segments:
            self.segments[p] /= t_step
            # The wavespeed is adjusted to compensate the truncation error
            e = int(self.segments[p])-self.segments[p] # truncation error
            self.a[p] = self.a[p]/(1 + e/self.segments[p])
            self.segments[p] = int(self.segments[p])

    def define_mesh(self):
        '''
        This function should be called only after defining the segments
        for each pipe in the network
        '''

        G = self.network
        
        # The segmented MOC-mesh graph is generated
        self.segmented_network = nx.Graph() 
        
        # The MOC-mesh graph will be traversed from a boundary node
        #   Because of the nature of the WDS it is always guaranteed
        #   to have a boundary node in the model.
        boundary_node = None

        # parfor
        # nb : Node at the beginning of the edge
        # ne : Node at the end of the edge
        for i, nb in enumerate(G):
            # A boundary node is chosen
            # It doesn't matter if there is a race condition
            if not boundary_node:
                if G.degree(nb) == 1:
                    boundary_node = nb
            
            for neighbor in G[nb]:
                for p in G[nb][neighbor]:
                    if G[nb][neighbor][p]['type'] == 'Pipe':
                        s_p = self.segments[p] # segments in p
                    else:
                        s_p = 0

                    n1 = nb

                    # Internal nodes are created (ni)
                    for j in range(s_p):
                        # Internal nodes labeled with k \in {-1, 1} are 
                        #   ghost nodes

                        k = -1 if j == s_p-1 else j+1

                        # 'initial_node.k.end_node'
                        ni = nb + '.' + str(k) + '.' + neighbor
                        self.segmented_network.add_edge(n1, ni)
                        n1 = ni

                    self.segmented_network.add_edge(n1, neighbor)

        # The undirected graph of the network is traversed using DFS
        #   DFS. The points in the MOC-mesh are allocated according
        #   to the DFS traversal. This is done to guarantee
        #   locality in memory and to minimize memory coalescing

        dfs = nx.dfs_preorder_nodes(self.segmented_network, source=boundary_node)
        
        # parfor
        for i, node in enumerate(dfs):
            self.order[node] = i + 1

    def write_mesh(self):
        '''
        This function should only be called after defining the mesh
        '''
        G = self.segmented_network
        # Network is stored in METIS format
        if G:
            with open(self.fname + '.graph', 'w') as f:
                f.write("%d %d\n" % (len(G), len(G.edges())))
                for i, node in enumerate(self.order):
                    fline = "" # file content
                    for neighbor in G[node]:
                        fline += "%d " % self.order[neighbor]
                    fline += '\n'
                    f.write(fline)