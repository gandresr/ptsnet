import wntr
import networkx as nx 
import matplotlib.pyplot as plt
import subprocess
from time import time
import numpy as np

# CUDA
# import pycuda.autoinit
# import pycuda.driver as drv
# from pycuda.compiler import SourceModule

class CUDA_solver:
    def __init__(self):
        kernels = []
    
    # def add_kernel(self, kernel):
    #     kernels.append(SourceModule(kernel))

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
        T: total time of simulation in seconds
        '''
        self.network = network
        self.interior_properties = None
        self.bc_properties = None
        self.junction_properties = None
        # Results
        m = int(T/network.dt)
        n = len(network.segmented_network)
        # Steady-state results
        self.ss_results = None
        self.H = np.zeros((m,n))
        self.Q = np.zeros((m,n))

    def define_initial_conditions(self):
        self.ss_results = wntr.sim.EpanetSimulator(self.network.wn).run_sim()
        for i, node in enumerate(self.network.order):
            if '.' in node: # Internal node
                labels = node.split('.') # [n1, k, n2]
                n1 = labels[0]
                n2 = labels[2]
                k = abs(int(labels[1]))
                p = self.network.get_pipe(n1, n2)
                
                head_1 = float(self.ss_results.node['head'][n2])
                head_2 = float(self.ss_results.node['head'][n1])
                hl = head_1 - head_2
                L = self.network.wn.get_link(p).length
                dx = k * L / self.network.segments[p]
                self.H[0,i] = head_1 - (hl*(1 - dx/L))
                self.Q[0,i] = float(self.ss_results.link['flowrate'][p])
            else: # Junction
                self.H[0,i] = float(self.ss_results.node['head'][node])
                self.Q[0,i] = float(self.ss_results.node['demand'][node])

    def define_MP_properties(self):
        '''
        In order to exploit the cache in the 
        shared memory scheme, tables are defined
        to be allocated and used locally by each
        thread
        '''
        pass

    def define_MPI_properties(self):
        pass
        


class MOC_network:
    def __init__(self, input_file):
        self.fname = input_file[:input_file.find('.inp')]
        self.wn = wntr.network.WaterNetworkModel(input_file)
        self.network = self.wn.get_graph()
        self.segmented_network = None
        self.a = {} # Wavespeed values
        self.dt = None
        self.hydraulic_model = wntr.sim.hydraulics.HydraulicModel(self.wn)
        # Segments are only defined for pipes
        self.segments = self.wn.query_link_attribute('length')
        self.order = {} # Ordered nodes

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
        self.dt = t_step

        # The number of segments is defined
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
                    for j in range(s_p-1):
                        # Internal nodes labeled with k \in {-s_p, 1} are 
                        #   ghost nodes

                        k = -(j+1) if j == s_p-2 else j+1

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

    def define_partitions(self, k):
        result = subprocess.call(['./kaffpa', self.fname + '.graph', '--k=' + str(k), '--preconfiguration=strong'])
        self.partitions = list(map(int, open('tmppartition' + str(k)).read()[:-1].split('\n')))

    def get_processor(self, node):
        return self.partitions[self.order[node]-1]
    
    def get_pipe(self, n1, n2):
        G = self.network
        for p in G[n1][n2]:
            return p

# class Grid_network:
#     '''
#     This class allows to generate networks with Grid topology 
#     and 2 boundary conditions, specificaly, one source reservoir
#     upstream and one valve downstream.

#     The topology of the network is created using networkx
#     and then, an MOC_network is created using WNTR and assigning
#     random parameters for (lengths, )
#     '''
#     def __init__(self, n, m):
