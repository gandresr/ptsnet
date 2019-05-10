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
        self.point_kernel = SourceModule('''
        __global__ void point_step(
            float * HH,
            float * QQ,
            float * Q1_point, 
            float * Q2_point, 
            float * H1_point, 
            float * H2_point, 
            float * wavespeed_point,
            float * D_point,
            float * frictionfact_point,
            float * dx_point,
            float * A_point)
        {
            const int i = threadIdx.x;
            float B, R, Cp, Cm, Bp, Bm;

            B = wavespeed[i]/(9.81*A_point[i]);
            R = frictionfact[i]*dx_point[i]/(2*9.81*D_point[i]*A_point[i]*A_point[i]);
            Cp = H1_point[i] + B*Q1_point[i];
            Cm = H2_point[i] - B*Q2_point[i];
            Bp = B + R*abs(Q1_point[i]);
            Bm = B + R*abs(Q2_point[i]);
            HH_points[i] = (Cp*Bm + Cm*Bp)/(Bp + Bm);
            QQ_points[i] = (Cp - Cm)/(Bp + Bm);
        }
        ''')

        # self.junction_kernel = SourceModule('''
        #
        # ''')

        self.valve_kernel = SourceModule('''
        __global__ void valve_step(
            float * H1_valve, 
            float * Q1_valve, 
            float * H0_valve, 
            float * Q0_valve, 
            float * setting, 
            float * wavespeed_valve, 
            float * D_valve, 
            float * frictionfact_valve, 
            float * dx_valve,
            float * area_valve
        )
            float B, R, Cv, Cp, Bp
            B = wavespeed_valve[i]/(9.81*area_valve[i])
            R = f*dx/(2*g*d*area_valve*area_valve)
            Cv = (Q0*tau)**2/(2*H0)
            Cp = H1 + B*Q1
            Bp = B + R*abs(Q1)
            QQ = -Bp*Cv + ((Bp*Cv)**2 + 2*Cv*Cp)**0.5
            HH = Cp - Bp*QQ
            return HH, QQ
        
        ''')
    
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
        
        self.network = network

        # Dimension of arrays is m x n
        m = T
        n = len(network.segmented_network)

        ## Pipe table
        # | wavespeeds | friction_factor | lengths | diameter | area
        self.pipe_properties = np.zeros((len(network.wavespeeds),5))
        ## Junction table
        # | demand
        self.junction_properties = np.zeros((len(network.network),1))
        ## Valves setting table
        # If file is not specified, then always open, i.e., setting = 1
        self.valve_settings = np.ones((T,self.network.wn.num_valves))

        # Steady-state results
        self.ss_results = None
        self.H = np.zeros((m,n))
        self.Q = np.zeros((m,n))

    def define_valve_setting(self, valve_id, valve_file):
        '''
        The valve_file has to be a CSV file
        '''
        settings = open(valve_file).read().split('\n')
        if ',' not in settings[0]: # Only one entry per line
            T = min(len(settings), len(self.valve_settings))
            for t in range(T):
                i = self.network.valves_order[valve_id]
                self.valve_settings[t, i-1] = settings[t] # Remember that orders start in 1

    def define_properties(self):
        '''
        In order to exploit the cache in the 
        shared memory scheme, tables are defined
        to be allocated and used locally by each
        thread
        '''
        # Pipe properties
        for i, p in enumerate(self.network.wavespeeds):
            # (0) wavespeed
            # (1) friction_factor
            # (2) length
            # (3) diameter
            # (4) area
            self.pipe_properties[i,0] = self.network.wavespeeds[p]
            self.pipe_properties[i,1] = float(self.ss_results.link['frictionfact'][p])
            self.pipe_properties[i,2] = self.network.wn.get_link(p).length
            self.pipe_properties[i,3] = self.network.wn.get_link(p).diameter
            self.pipe_properties[i,4] = np.pi*(self.pipe_properties[i][3])**2/4

        # Junction properties
        # Note: notice that junctions with negative demands are source reservoirs
        for i, junction in enumerate(self.network.network):
            # (0) demand
            self.junction_properties[i, 0] = self.ss_results.node['demand'][junction]

    def define_MPI(self):
        pass
        
    def define_initial_conditions(self):
        self.ss_results = wntr.sim.EpanetSimulator(self.network.wn).run_sim()
        for i, node in enumerate(self.network.nodes_order):
            if '.' in node: # Points
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
            else: # Junctions
                self.H[0,i] = float(self.ss_results.node['head'][node])
                self.Q[0,i] = float(self.ss_results.node['demand'][node])

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
                self.pipes_order[p] = i + 1 # +1 to be consistent with nodes_order
                i += 1
            elif self.wn.get_link(p).link_type == 'Valve':
                self.valves_order[p] = j + 1
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
            print(i,type(node))
            self.nodes_order[node] = i + 1

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
                        fline += "%d " % self.nodes_order[neighbor]
                    fline += '\n'
                    f.write(fline)

    def define_partitions(self, k):
        result = subprocess.call(['./kaffpa', self.fname + '.graph', '--k=' + str(k), '--preconfiguration=strong'])
        self.partitions = np.array(list(map(int, open('tmppartition' + str(k)).read()[:-1].split('\n'))))

    def get_processor(self, node):
        return self.partitions[self.nodes_order[node]-1]
    
    def get_pipe_name(self, n1, n2):
        if n2 not in self.network[n1]:
            return None
        for p in self.network[n1][n2]:
            return p
