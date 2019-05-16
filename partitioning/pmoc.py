import wntr
import networkx as nx 
import matplotlib.pyplot as plt
import subprocess
from time import time
import numpy as np


class Wall_clock:
    def __init__(self):
        self.clk = time()
    
    def tic(self):
        self.clk = time()
    
    def toc(self):
        print('Elapsed time: %f seconds' % (time() - self.clk))

class MOC_simulation:
    '''
    Here all the tables and properties required to
    run a MOC simulation are defined. Tables for
    simulations with CUDA, OpenMP, and OpenMPI are
    created
    '''
    def __init__(self, network, T):
        '''
        Requires an MOC_network
        T: total time steps
        '''
        pass

class MOC_network:
    def __init__(self, input_file):
        '''
        * The network graph is generated by WNTR
        * The MOC_network is a segmented network that includes 
            new nodes between pipes which are denominated points
        * The nodes in the network graph are denominated junctions
        '''
        self.fname = input_file[:input_file.find('.inp')]
        self.wn = wntr.network.WaterNetworkModel(input_file)
        self.network = self.wn.get_graph()
        self.segmented_network = None
        self.wavespeeds = {}
        self.valve_nodes = []
        self.dt = None
        
        # Segments are only defined for pipes
        self.segments = self.wn.query_link_attribute('length')
        
        ## Order dictionaries
        # All the orders start in 1
        self.nodes_order = {} # Ordered nodes (nodes in segmented graph)
        # Ordered pipes and valves (in WNTR graph)
        self.pipes_order = {}
        self.valves_order = {}
        
        # Nodes connected to valves
        self.valve_nodes = [valve[1].end_node_name for valve in self.wn.valves()] + \
            [valve[1].start_node_name for valve in self.wn.valves()]
        
        i = 0; j = 0
        for (n1, n2) in self.network.edges():
            p = self.get_pipe_name(n1, n2)
            if self.wn.get_link(p).link_type == 'Pipe':
                self.pipes_order[p] = i
                i += 1
            elif self.wn.get_link(p).link_type == 'Valve':
                self.valves_order[p] = j
                j += 1
                
        
        self.partitions = None

    def define_wavespeeds(self, default_wavespeed = 1200, wavespeed_file = None):
        if wavespeed_file:
            '''
            CSV file only
            '''
            f = open(wavespeed_file).read().split('\n')
            for p, a_p in map(lambda x : x.split(','), f):
                self.wavespeeds[p] = float(a_p)
        else:
            for p, data in self.wn.links():
                if data.link_type == 'Pipe':
                    self.wavespeeds[p] = default_wavespeed

    def define_segments(self, dt):
        # Get the maximum time steps for each pipe
        for p in self.segments:
            self.segments[p] /= self.wavespeeds[p]
        
        # Maximum dt in the system to capture waves in all pipes
        max_dt = self.segments[min(self.segments, key=self.segments.get)]

        # Desired dt < max_dt ?
        t_step = min(dt, max_dt)
        self.dt = t_step

        # The number of segments is defined
        for p in self.segments:
            self.segments[p] /= t_step
            # The wavespeed is adjusted to compensate the truncation error
            e = int(self.segments[p])-self.segments[p] # truncation error
            self.wavespeeds[p] = self.wavespeeds[p]/(1 + e/self.segments[p])
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
            if not boundary_node:
                if G.degree(nb) == 1:
                    boundary_node = nb
            
            for neighbor in G[nb]:
                for p in G[nb][neighbor]:
                    n1 = nb
                    link = self.wn.get_link(p)
                    if link.link_type == 'Pipe':
                        s_p = self.segments[p] # segments in p
                    elif link.link_type == 'Valve':
                        s_p = 0

                    # Points are created (ni)
                    for j in range(s_p-1):
                        # Points labeled with k \in {-s_p, 1} are 
                        #   ghost nodes

                        k = -j if j == s_p-2 else j

                        # 'initial_node.k.end_node'
                        ni = nb + '.' + str(k) + '.' + neighbor
                        self.segmented_network.add_edge(n1, ni)
                        n1 = ni

                    self.segmented_network.add_edge(n1, neighbor)

        # The undirected graph of the network is traversed using DFS
        #   DFS. The points in the MOC-mesh are allocated according
        #   to the DFS traversal. This is done to guarantee
        #   locality in memory and to minimize memory coalescing

        # dfs = nx.dfs_preorder_nodes(self.segmented_network, source=boundary_node)
        
        # parfor
        for i, node in enumerate(self.segmented_network):
            self.nodes_order[node] = i

    def write_mesh(self):
        '''
        This function should only be called after defining the mesh
        '''
        G = self.segmented_network
        # Network is stored in METIS format
        if G:
            with open(self.fname + '.graph', 'w') as f:
                f.write("%d %d\n" % (len(G), len(G.edges())))
                for i, node in enumerate(self.nodes_order):
                    fline = "" # file content
                    for neighbor in G[node]:
                        fline += "%d " % (self.nodes_order[neighbor] + 1)
                    fline += '\n'
                    f.write(fline)

    def define_partitions(self, k):
        result = subprocess.call(['./kaffpa', self.fname + '.graph', '--k=' + str(k), '--preconfiguration=strong'])
        self.partitions = np.array(list(map(int, open('tmppartition' + str(k)).read()[:-1].split('\n'))))

    def get_processor(self, node):
        return self.partitions[self.nodes_order[node]]
    
    def get_pipe_name(self, n1, n2):
        if n2 not in self.network[n1]:
            return None
        for p in self.network[n1][n2]:
            return p
