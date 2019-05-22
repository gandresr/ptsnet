import wntr
import networkx as nx 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from numba import njit 

from time import time
import subprocess

# Temporal
import enum

class MOC_simulation:
    # Enumerators

    class Node(enum.Enum):
        node_type = 0 # {none, reservoir, junction, end, valve_a, valve_b}
        link_id = 1 # {none (-1), link_id, valve_id}
        processor = 2
        is_ghost = 3
  
    class Pipe(enum.Enum):
        node_a = 0
        node_b = 1
        diameter = 2
        area = 3
        wavespeed = 4
        ffactor = 5
        length = 6
        dx = 7

    class Valve(enum.Enum):
        node_a = 0
        node_b = 1
      
    class node_types(enum.Enum):
        none = 0
        reservoir = 1
        junction = 2
        interior = 3
        valve_a = 4
        valve_b = 5

    '''
    Here all the tables and properties required to
    run a MOC simulation are defined. Tables for
    simulations in parallel are created

    In the meantime:
    * valves are not valid in general junctions
    * it is not possible to connect one valve to another
    * valves should be 100% open for initial conditions
    '''
    def __init__(self, network, T):
        '''
        Requires an MOC_network
        T: total time steps
        '''
        self.moc_network = network
        self.time_steps = T

        # Simulation results
        self.steady_state_sim = wntr.sim.EpanetSimulator(network.wn).run_sim()
        self.flow_results = np.zeros( (len(network.mesh), T) )
        self.head_results = np.zeros( (len(network.mesh), T) )
        self.upstream_flow_results = []
        self.downstream_flow_results = []
        
        # a[a[:,self.Node.processor.value].argsort()] - Sort by processor
        self.nodes = np.zeros((len(network.mesh), len(self.Node)), dtype=int)
        self.upstream_pipes = []
        self.downstream_pipes = []
        self.upstream_nodes = []
        self.downstream_nodes = []

        for i in range(len(network.mesh)):
            self.upstream_pipes.append( [] )
            self.downstream_pipes.append( [] )
            self.upstream_nodes.append( [] )
            self.downstream_nodes.append( [] )
        
        self.junction_ids = {}
        self.junction_names = []

        self.pipes = np.zeros((network.wn.num_pipes, len(self.Pipe)))
        self.valves = np.zeros((network.wn.num_valves, len(self.Valve)))
        self._define_properties()

        # Simulation inputs
        self._define_initial_conditions()
        self.valve_settings = np.ones( (network.wn.num_valves, T) )

    def run_step(self, t, thread_id, N):
        g = 9.81
        for i in range(thread_id, thread_id+N):
            node_type = self.nodes[i, self.Node.node_type.value]
            node_name = self.moc_network.node_names[i]
            u_pipes = self.upstream_pipes[i]
            d_pipes = self.downstream_pipes[i]
            u_nodes = self.upstream_nodes[i]
            d_nodes = self.downstream_nodes[i]

            if node_type == self.node_types.reservoir.value:
                pass
            elif node_type == self.node_types.interior.value:
                if len(u_pipes) != 1 or len(d_pipes) != 1:
                    raise Exception("There is an error with the data structures")

                u_node_id = self.moc_network.node_ids[u_nodes[0]]
                d_node_id = self.moc_network.node_ids[d_nodes[0]]
                pipe_id = self.nodes[i, self.Node.link_id.value]

                u_node_type = self.nodes[u_node_id, self.Node.node_type.value]
                d_node_type = self.nodes[d_node_id, self.Node.node_type.value]
                
                # Extract heads
                H1 = self.head_results[u_node_id, t-1] 
                H2 = self.head_results[d_node_id, t-1]

                Q1 = None; Q2 = None
                # Extract flows
                if u_node_type == self.node_types.junction.value:
                    j = self.junction_ids[u_nodes[0]]
                    Q1 = self.downstream_flow_results[j][self.downstream_nodes[u_node_id].index(node_name)][t-1]
                else:
                    Q1 = self.flow_results[u_node_id, t-1]
                if d_node_type == self.node_types.junction.value:
                    j = self.junction_ids[d_nodes[0]]
                    Q2 = self.upstream_flow_results[j][self.upstream_nodes[d_node_id].index(node_name)][t-1]
                else:
                    Q2 = self.flow_results[d_node_id, t-1]

                wavespeed = self.pipes[pipe_id, self.Pipe.wavespeed.value]
                area = self.pipes[pipe_id, self.Pipe.area.value]
                ffactor = self.pipes[pipe_id, self.Pipe.ffactor.value]
                diameter = self.pipes[pipe_id, self.Pipe.diameter.value]
                dx = self.pipes[pipe_id, self.Pipe.dx.value]

                B = wavespeed/(g*area)
                R = ffactor*dx/(2*g*diameter*area**2)

                Cp = H1 + B*Q1
                Cm = H2 - B*Q2
                Bp = B + R*abs(Q1)
                Bm = B + R*abs(Q2)
                
                # Save head and flow results at node
                self.head_results[i, t] = (Cp*Bm + Cm*Bp)/(Bp + Bm)
                self.flow_results[i, t] = (Cp - Cm)/(Bp + Bm) 

            elif node_type == self.node_types.junction.value:

                sc = 0
                sb = 0
                
                Cp = np.zeros(len(u_pipes))
                Bp = np.zeros_like(Cp)
                
                Cm = np.zeros(len(d_pipes))
                Bm = np.zeros_like(Cm)

                for j, u_pipe in enumerate(u_pipes):
                    u_node = u_nodes[j]

                    u_pipe_id = self.moc_network.pipe_ids[u_pipe]
                    u_node_id = self.moc_network.node_ids[u_node]

                    wavespeed = self.pipes[u_pipe_id, self.Pipe.wavespeed.value]
                    area = self.pipes[u_pipe_id, self.Pipe.area.value]
                    ffactor = self.pipes[u_pipe_id, self.Pipe.ffactor.value]
                    diameter = self.pipes[u_pipe_id, self.Pipe.diameter.value]
                    dx = self.pipes[u_pipe_id, self.Pipe.dx.value]

                    B = wavespeed/(g*area)
                    R = ffactor*dx/(2*g*diameter*area**2)
                    H1 = self.head_results[u_node_id, t-1]
                    Q1 = self.flow_results[u_node_id, t-1]

                    Cp[j] = H1 + B*Q1
                    Bp[j] = B + R*abs(Q1)
                    sc += Cp[j]/Bp[j]
                    sb += 1/Bp[j]

                for j, d_pipe in enumerate(d_pipes):
                    d_node = d_nodes[j]

                    d_pipe_id = self.moc_network.pipe_ids[d_pipe]
                    d_node_id = self.moc_network.node_ids[d_node]

                    wavespeed = self.pipes[d_pipe_id, self.Pipe.wavespeed.value]
                    area = self.pipes[d_pipe_id, self.Pipe.area.value]
                    ffactor = self.pipes[d_pipe_id, self.Pipe.ffactor.value]
                    diameter = self.pipes[d_pipe_id, self.Pipe.diameter.value]
                    dx = self.pipes[d_pipe_id, self.Pipe.dx.value]

                    B = wavespeed/(g*area)
                    R = ffactor*dx/(2*g*diameter*area**2)
                    H1 = self.head_results[d_node_id, t-1]
                    Q1 = self.flow_results[d_node_id, t-1]
                    Cm[j] = H1 - B*Q1
                    Bm[j] = B + R*abs(Q1)
                    sc += Cm[j]/Bm[j]
                    sb += 1/Bm[j]
                
                # Update new head at node
                HH = sc/sb
                self.head_results[i, t] = HH

                node_name = self.moc_network.node_names[i]
                junction_id = self.junction_ids[node_name]

                # Update new flows at node
                for j in range(len(u_pipes)):
                    self.upstream_flow_results[junction_id][j][t] = (Cp[j] - HH)/Bp[j]
                for j in range(len(d_pipes)):
                    self.downstream_flow_results[junction_id][j][t] = (HH - Cm[j])/Bm[j]

    def _define_properties(self):
        self._define_pipes()
        self._define_nodes()
        self._define_valves()

    def _define_nodes(self):
        i = 0 # Index to count junctions
        for node, node_id in self.moc_network.node_ids.items():

            ## TYPE & LINK_ID ARE DEFINED
            # ----------------------------------------------------------------------------------------------------------------

            # Remember that mesh is an undirected networkx Graph
            neighbors = list(self.moc_network.mesh.neighbors(node))
            if node in self.moc_network.wn.reservoir_name_list:
                # Check if node is reservoir node
                self.nodes[node_id, self.Node.node_type.value] = self.node_types.reservoir.value
                self.nodes[node_id, self.Node.link_id.value] = -1
            # Check if the node belongs to a valve
            if self.nodes[node_id, self.Node.node_type.value] == self.node_types.none.value: # Type not defined yet
                if not '.' in node:
                    for valve, valve_id in self.moc_network.valve_ids.items():
                        link = self.moc_network.wn.get_link(valve)
                        start = link.start_node_name
                        end = link.end_node_name
                        if node == start:
                            self.nodes[node_id, self.Node.node_type.value] = self.node_types.valve_a.value
                        elif node == end:
                            self.nodes[node_id, self.Node.node_type.value] = self.node_types.valve_b.value
                        if node in (start, end):
                            # link_id is associated to a valve
                            self.nodes[node_id, self.Node.link_id.value] = valve_id
                            break
            if self.nodes[node_id, self.Node.node_type.value] == self.node_types.none.value: # Type not defined yet
                # Node is considered a junction if there is more than one pipe attached to it is not a valve or reservoir
                if len(neighbors) > 1:
                    if '.' in node: # interior points
                        self.nodes[node_id, self.Node.node_type.value] = self.node_types.interior.value
                        labels = node.split('.') # [n1, k, n2, p]
                        n1 = labels[0]
                        n2 = labels[2]
                        pipe = labels[3]
                        self.nodes[node_id, self.Node.link_id.value] = self.moc_network.pipe_ids[pipe]
                    else:
                        self.nodes[node_id, self.Node.node_type.value] = self.node_types.junction.value
                        self.upstream_flow_results.append( [] )
                        self.downstream_flow_results.append( [] )
                        self.nodes[node_id, self.Node.link_id.value] = -1

            # ----------------------------------------------------------------------------------------------------------------

            ## PROCESSOR & IS_GHOST ARE DEFINED
            # ----------------------------------------------------------------------------------------------------------------

            self.nodes[node_id, self.Node.processor.value] = self.moc_network.get_processor(node)
            self.nodes[node_id, self.Node.is_ghost.value] = (self.moc_network.separator[node_id] == self.moc_network.num_processors)

            # ----------------------------------------------------------------------------------------------------------------
    
            ## NODE UPSTREAM & DOWNSTREAM PIPES INFORMATION
            # ----------------------------------------------------------------------------------------------------------------

            if '.' in node:
                labels = node.split('.')
                n1 = labels[0]
                k = int(labels[1])
                n2 = labels[2]
                pipe = labels[3]
                segments = int(labels[4])
                if k == 0:
                    self.upstream_pipes[node_id].append(pipe)
                    self.downstream_pipes[node_id].append(pipe)
                    self.upstream_nodes[node_id].append(n1)
                    self.downstream_nodes[node_id].append(
                        n1 + '.1.' + n2 + '.' + pipe + '.' + str(segments))
                elif k == segments-2: # last interior point in pipe
                    self.upstream_pipes[node_id].append(pipe)
                    self.downstream_pipes[node_id].append(pipe)
                    self.upstream_nodes[node_id].append(
                        n1 + '.' + str(abs(k) - 1) + '.' + n2 + '.' + pipe + '.' + str(segments))
                    self.downstream_nodes[node_id].append(n2)
                else:
                    self.upstream_pipes[node_id].append(pipe)
                    self.downstream_pipes[node_id].append(pipe)
                    self.upstream_nodes[node_id].append(
                        n1 + '.' + str(abs(k) - 1) + '.' + n2 + '.' + pipe + '.' + str(segments))
                    self.downstream_nodes[node_id].append(
                        n1 + '.' + str(abs(k) + 1) + '.' + n2 + '.' + pipe + '.' + str(segments))
            else:
                if self.nodes[node_id, self.Node.node_type.value] == self.node_types.junction.value:
                    self.junction_ids[node] = i
                    self.junction_names.append(node)
                    i += 1
                for n in neighbors:
                    if '.' in n:
                        labels_n = n.split('.')
                        k = int(labels_n[1])
                        pipe = labels_n[3]
                        if k == 0:
                            self.downstream_pipes[node_id].append(pipe)
                            self.downstream_nodes[node_id].append(n)
                            if len(neighbors) > 2:
                                self.downstream_flow_results[-1].append( np.zeros( self.time_steps ) )
                        else:
                            self.upstream_pipes[node_id].append(pipe)
                            self.upstream_nodes[node_id].append(n)
                            if len(neighbors) > 2:
                                self.upstream_flow_results[-1].append( np.zeros( self.time_steps ) )
                    else:
                        pipe = self.moc_network.get_pipe_name(n, node)
                        if pipe == None:
                            pipe = self.moc_network.get_pipe_name(node, n)
                            if self.nodes[node_id, self.Node.node_type.value] == self.node_types.valve_a.value:
                                self.downstream_pipes[node_id].append(pipe)
                            elif self.nodes[node_id, self.Node.node_type.value] == self.node_types.junction.value:
                                self.downstream_pipes[node_id].append(pipe)
                            self.downstream_nodes[node_id].append(n)
                        else:
                            if self.nodes[node_id, self.Node.node_type.value] == self.node_types.valve_b.value:
                                self.upstream_pipes[node_id].append(pipe)
                            elif self.nodes[node_id, self.Node.node_type.value] == self.node_types.junction.value:
                                self.upstream_pipes[node_id].append(pipe)
                            self.upstream_nodes[node_id].append(n)
           
            # ----------------------------------------------------------------------------------------------------------------

    def _define_pipes(self):
        for pipe, pipe_id in self.moc_network.pipe_ids.items():
            link = self.moc_network.wn.get_link(pipe)
            self.pipes[pipe_id, self.Pipe.node_a.value] = self.moc_network.node_ids[link.start_node_name]
            self.pipes[pipe_id, self.Pipe.node_b.value] = self.moc_network.node_ids[link.end_node_name]
            diameter = link.diameter
            self.pipes[pipe_id, self.Pipe.diameter.value] = diameter
            self.pipes[pipe_id, self.Pipe.area.value] = np.pi*diameter**2/4
            self.pipes[pipe_id, self.Pipe.wavespeed.value] = self.moc_network.wavespeeds[pipe]
            self.pipes[pipe_id, self.Pipe.ffactor.value] = float(self.steady_state_sim.link['frictionfact'][pipe])
            self.pipes[pipe_id, self.Pipe.length.value] = link.length
            self.pipes[pipe_id, self.Pipe.dx.value] = link.length / self.moc_network.segments[pipe]

    def _define_valves(self):
        for valve, valve_id in self.moc_network.valve_ids.items():
            link = self.moc_network.wn.get_link(valve)
            self.valves[valve_id, self.Valve.node_a.value] = link.start_node_name
            self.valves[valve_id, self.Valve.node_b.value] = link.end_node_name

    def define_valve_setting(self, valve_name, valve_file):
        '''
        The valve_file has to be a file with T <= self.time_steps lines 
        The i-th of the file has the value of the valve setting at 
        the i-th time step. If the valve setting is not defined in the file
        for a certain time step, it is assumed that the valve will be
        fully open at that time step.
        '''
        settings = np.loadtxt(valve_file, dtype=float)
        valve_id = self.moc_network.valve_ids[valve_name]

        T = min(len(settings), self.time_steps)
            
        for t in range(T):
            self.valve_settings[valve_id, t] = settings[t]

    def _define_initial_conditions(self):
        for node, node_id in self.moc_network.node_ids.items():
            if '.' in node: # interior points
                labels = node.split('.') # [n1, k, n2, p]
                n1 = labels[0]
                k = abs(int(labels[1]))
                n2 = labels[2]
                pipe = labels[3]
                
                head_1 = float(self.steady_state_sim.node['head'][n2])
                head_2 = float(self.steady_state_sim.node['head'][n1])
                hl = head_1 - head_2
                L = self.moc_network.wn.get_link(pipe).length
                dx = k * L / self.moc_network.segments[pipe]
                
                self.head_results[node_id, 0] = head_1 - (hl*(1 - dx/L))
                self.flow_results[node_id, 0] = float(self.steady_state_sim.link['flowrate'][pipe])
            else:
                head = float(self.steady_state_sim.node['head'][node])
                for j, neighbor in enumerate(self.upstream_nodes[node_id]):
                    link_name = None
                    neighbor_id = self.moc_network.node_ids[neighbor]
                    if self.nodes[neighbor_id, self.Node.node_type.value] in (self.node_types.valve_a, self.node_types.valve_b):
                        link_name = self.moc_network.valve_names[self.nodes[neighbor_id, self.Node.link_id.value]]
                    else:
                        idx = int(self.nodes[neighbor_id, self.Node.link_id.value])
                        link_name = self.moc_network.pipe_names[idx]
                    
                    if len(list(self.moc_network.mesh.neighbors(node))) > 2:
                        junction_id = self.junction_ids[node]
                        self.upstream_flow_results[junction_id][j][0] = float(
                            self.steady_state_sim.link['flowrate'][link_name])
                for j, neighbor in enumerate(self.downstream_nodes[node_id]):
                    link_name = None
                    neighbor_id = self.moc_network.node_ids[neighbor]
                    if self.nodes[neighbor_id, self.Node.node_type.value] in (self.node_types.valve_a, self.node_types.valve_b):
                        link_name = self.moc_network.valve_names[self.nodes[neighbor_id, self.Node.link_id.value]]
                    else:
                        link_name = self.moc_network.pipe_names[int(self.nodes[neighbor_id, self.Node.link_id.value])]
                    
                    if len(list(self.moc_network.mesh.neighbors(node))) > 2:
                        junction_id = self.junction_ids[node]
                        self.downstream_flow_results[junction_id][j][0] = float(
                            self.steady_state_sim.link['flowrate'][link_name])

                self.head_results[node_id, 0] = head

class MOC_network:
    def __init__(self, input_file):
        '''
        * The network graph is generated by WNTR
        * The MOC_network is a segmented network that includes 
            new nodes between pipes which are denominated interior points
        * The nodes in the network graph are denominated junctions
        '''
        self.fname = input_file[:input_file.find('.inp')]
        self.wn = wntr.network.WaterNetworkModel(input_file)
        self.network = self.wn.get_graph()
        self.mesh = None
        self.wavespeeds = {}
        self.dt = None
        self.num_processors = None # number of processors

        # Number of segments are only defined for pipes
        self.segments = self.wn.query_link_attribute('length')
        
        # Ids for nodes, pipes, and valves
        self.node_ids = {}
        self.node_names = []
        self.pipe_ids = {}
        self.pipe_names = []
        self.valve_ids = {}
        self.valve_names = []
                
        self.partition = None
        self.separator = None
   
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
        self.mesh = nx.Graph() 
        
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

                    # interior points are created (ni)
                    for j in range(s_p-1):
                        # 'initial_node.k.end_node'
                        ni = nb + '.' + str(j) + '.' + neighbor + '.' + p + '.' + str(self.segments[p])
                        self.mesh.add_edge(n1, ni)
                        n1 = ni

                    self.mesh.add_edge(n1, neighbor)
        
        # parfor
        self._define_ids()

    def _define_ids(self):
        for i, node in enumerate(self.mesh):
            self.node_ids[node] = i
            self.node_names.append(node)
        i = 0; j = 0
        for (n1, n2) in self.network.edges():
            p = self.get_pipe_name(n1, n2)
            link = self.wn.get_link(p)
            if link.link_type == 'Pipe':
                self.pipe_ids[p] = i
                self.pipe_names.append(p)
                i += 1
            elif link.link_type == 'Valve':
                self.valve_ids[p] = j
                self.valve_names.append(p)
                j += 1

    def write_mesh(self):
        '''
        This function should only be called after defining the mesh
        '''
        G = self.mesh
        # Network is stored in METIS format
        if G:
            with open(self.fname + '.graph', 'w') as f:
                f.write("%d %d\n" % (len(G), len(G.edges())))
                for node in self.node_ids:
                    fline = "" # file content
                    for neighbor in G[node]:
                        fline += "%d " % (self.node_ids[neighbor] + 1)
                    fline += '\n'
                    f.write(fline)

    def define_partitions(self, k):
        script = './parHIP/kaffpa'
        subprocess.call([
            script, self.fname + '.graph', 
            '--k=' + str(k), 
            '--preconfiguration=strong', 
            '--output_filename=partitionings/p%d' % k])

        if k == 2:
            script = './parHIP/node_separator'
        else:
            script = './parHIP/partition_to_vertex_separator'

        subprocess.call([
            script, self.fname + '.graph', 
            '--k=' + str(k), 
            '--input_partition=partitionings/p%d' % k, 
            '--output_filename=partitionings/s%d' % k])

        self.num_processors = k
        self.partition = np.loadtxt('partitionings/p%d' % k, dtype=int)
        self.separator = np.loadtxt('partitionings/s%d' % k, dtype=int)
        
    def get_processor(self, node):
        return self.partition[self.node_ids[node]]
    
    def get_pipe_name(self, n1, n2):
        try:
            for p in self.network[n1][n2]:
                return p
        except:
            return None

class Wall_clock:
    def __init__(self):
        self.clk = time()
    
    def tic(self):
        self.clk = time()
    
    def toc(self):
        print('Elapsed time: %f seconds' % (time() - self.clk))