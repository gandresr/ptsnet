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

    class Valve(enum.Enum):
        node_a = 0
        node_b = 1
      
    class node_type(enum.Enum):
        none = 0
        reservoir = 1
        junction = 2
        end = 3
        valve_a = 4
        valve_b = 5

    '''
    Here all the tables and properties required to
    run a MOC simulation are defined. Tables for
    simulations in parallel are created
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
        self.flow_results = np.empty( (len(network.mesh), T) )
        self.head_results = np.empty( (len(network.mesh), T) )
        self._define_initial_conditions()
        
        # a[a[:,self.Node.processor.value].argsort()] - Sort by processor
        self.nodes = np.empty((len(network.mesh), len(self.Node)))
        self.pipes_upstream = [[] for i in range(len(network.mesh))]
        self.pipes_downstream = [[] for i in range(len(network.mesh))]
        self.nodes_upstream = [[] for i in range(len(network.mesh))]
        self.nodes_downstream = [[] for i in range(len(network.mesh))]

        self.pipes = np.empty((network.wn.num_pipes, len(self.Pipe)))
        self.valves = np.empty((network.wn.num_valves, len(self.Valve)))
        self._define_properties()

        # Simulation inputs
        self.valve_settings = np.ones( (network.wn.num_valves, T) )

    def run_step(self):
        pass
    
    def _run_junction_step(self, i, np):
        '''
        '''
        for i in range(np):
            pipe_id = self.nodes[i, self.Node.link_id]
            wavespeed = self.pipes[pipe_id, self.Pipe.wavespeed]
            diameter = self.pipes[pipe_id, self.Pipe.diameter]
            area = self.pipes[pipe_id, self.Pipe.area]
            length = self.pipes[pipe_id, self.Pipe.length]
        
        def run_junction_bc(u_pipes, d_pipes, t):
            sc = 0
            sb = 0
            Cp = [0 for i in range(len(u_pipes))]
            Bp = [0 for i in range(len(u_pipes))]
            uQQ = [0 for i in range(len(u_pipes))]
            Cm = [0 for i in range(len(d_pipes))]
            Bm = [0 for i in range(len(d_pipes))]
            dQQ = [0 for i in range(len(d_pipes))]

            for i, p in enumerate(u_pipes['pipes']):
                A = pi*u_pipes['d'][i]**2/4
                g = 9.81
                B = u_pipes['a'][i]/(g*A)
                R = u_pipes['f'][i]*u_pipes['dx'][i]/(2*g*u_pipes['d'][i]*A**2)
                H1 = u_pipes['H1'][i]
                Q1 = u_pipes['Q1'][i]
                Cp[i] = H1 + B*Q1
                Bp[i] = B + R*abs(Q1)
                sc += Cp[i]/Bp[i]
                sb += 1/Bp[i]
            for i, p in enumerate(d_pipes['pipes']):
                A = pi*d_pipes['d'][i]**2/4
                g = 9.81
                B = d_pipes['a'][i]/(g*A)
                R = d_pipes['f'][i]*d_pipes['dx'][i]/(2*g*d_pipes['d'][i]*A**2)
                H1 = d_pipes['H1'][i]
                Q1 = d_pipes['Q1'][i]
                Cm[i] = H1 - B*Q1
                Bm[i] = B + R*abs(Q1)
                sc += Cm[i]/Bm[i]
                sb += 1/Bm[i]
            HH = sc/sb
            for i, p in enumerate(u_pipes['pipes']):
                uQQ[i] = (Cp[i] - HH)/Bp[i]
            for i, p in enumerate(d_pipes['pipes']):
                dQQ[i] = (HH - Cm[i])/Bm[i]

            return (HH, uQQ, dQQ)

    def _run_valve_step(self):
        pass

    def _run_source_step(self):
        pass
    
    def get_valve_curve(self, valve_name):
        pass

    def _define_properties(self):
        self._define_pipes()
        self._define_nodes()
        self._define_valves()

    def _define_nodes(self):
        '''
        In the meantime, valves are not valid in general junctions
        also it is not possible to connect one valve to another
        '''
        for node, node_id in self.moc_network.node_ids.items():

            ## TYPE & LINK_ID ARE DEFINED
            # ----------------------------------------------------------------------------------------------------------------

            # Remember that mesh is an undirected networkx Graph
            neighbors = list(self.moc_network.mesh.neighbors(node))
            if node in self.moc_network.wn.reservoir_name_list:
                # Check if node is reservoir node
                self.nodes[node_id, self.Node.node_type.value] = self.node_type.reservoir.value
                self.nodes[node_id, self.Node.link_id.value] = -1
            # Check if the node belongs to a valve
            if self.nodes[node_id, self.Node.node_type.value] == self.node_type.none.value: # Type not defined yet
                for valve, valve_id in self.moc_network.valve_ids.items():
                    link = self.moc_network.wn.get_link(valve)
                    start = link.start_node_name
                    end = link.end_node_name
                    if node == start:
                        self.nodes[node_id, self.Node.node_type.value] = self.node_type.valve_a.value
                        # link_id is associated to a valve
                        self.nodes[node_id, self.Node.link_id.value] = valve_id
                        break
                    elif node == end:
                        self.nodes[node_id, self.Node.node_type.value] = self.node_type.valve_b.value
                        # link_id is associated to a valve
                        self.nodes[node_id, self.Node.link_id.value] = valve_id
                        break
            if self.nodes[node_id, self.Node.node_type.value] == self.node_type.none.value: # Type not defined yet
                # Node is considered a junction if there is more than one pipe attached to it is not a valve or reservoir
                if len(neighbors) > 1:
                    self.nodes[node_id, self.Node.node_type.value] = self.node_type.junction.value
                    if '.' in node: # interior points
                        labels = node.split('.') # [n1, k, n2, p]
                        n1 = labels[0]
                        n2 = labels[2]
                        pipe = labels[3]
                        self.nodes[node_id, self.Node.link_id.value] = self.moc_network.pipe_ids[pipe]
                    else:
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
                    self.pipes_upstream[node_id].append(self.moc_network.pipe_ids[pipe])
                    self.pipes_downstream[node_id].append(self.moc_network.pipe_ids[pipe])
                    self.nodes_upstream[node_id].append(self.moc_network.node_ids[n1])
                    self.nodes_downstream[node_id].append(self.moc_network.node_ids[
                        n1 + '.1.' + n2 + '.' + pipe + '.' + str(segments)])
                elif k == segments-2: # last interior point in pipe
                    self.pipes_upstream[node_id].append(self.moc_network.pipe_ids[pipe])
                    self.pipes_downstream[node_id].append(self.moc_network.pipe_ids[pipe])
                    self.nodes_upstream[node_id].append(self.moc_network.node_ids[
                        n1 + '.' + str(abs(k) - 1) + '.' + n2 + '.' + pipe + '.' + str(segments)])
                    self.nodes_downstream[node_id].append(self.moc_network.node_ids[n2])
                else:
                    self.pipes_upstream[node_id].append(self.moc_network.pipe_ids[pipe])
                    self.pipes_downstream[node_id].append(self.moc_network.pipe_ids[pipe])
                    self.nodes_upstream[node_id].append(self.moc_network.node_ids[
                        n1 + '.' + str(abs(k) - 1) + '.' + n2 + '.' + pipe + '.' + str(segments)])
                    self.nodes_downstream[node_id].append(self.moc_network.node_ids[
                        n1 + '.' + str(abs(k) + 1) + '.' + n2 + '.' + pipe + '.' + str(segments)])
            else:
                neighbors = self.moc_network.mesh.neighbors(node)
                for n in neighbors:
                    if '.' in n:
                        labels_n = n.split('.')
                        k = int(labels_n[1])
                        pipe = labels_n[3]
                        if k == 0:
                            self.pipes_downstream[node_id].append(self.moc_network.pipe_ids[pipe])
                            self.nodes_downstream[node_id].append(self.moc_network.node_ids[n])
                        else:
                            self.pipes_upstream[node_id].append(self.moc_network.pipe_ids[pipe])
                            self.nodes_upstream[node_id].append(self.moc_network.node_ids[n])
                    else:
                        pipe = self.moc_network.get_pipe_name(n, node)
                        if pipe == None:
                            pipe = self.moc_network.get_pipe_name(node, n)
                            if self.nodes[node_id, self.Node.node_type.value] == self.node_type.valve_a.value:
                                self.pipes_downstream[node_id].append(self.moc_network.valve_ids[pipe])
                            else:
                                self.pipes_downstream[node_id].append(self.moc_network.pipe_ids[pipe])
                            self.nodes_downstream[node_id].append(self.moc_network.node_ids[n])
                        else:
                            if self.nodes[node_id, self.Node.node_type.value] == self.node_type.valve_b.value:
                                self.pipes_upstream[node_id].append(self.moc_network.valve_ids[pipe])
                            else:
                                self.pipes_upstream[node_id].append(self.moc_network.pipe_ids[pipe])
                            self.nodes_upstream[node_id].append(self.moc_network.node_ids[n])
           
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
        for node, idx in self.moc_network.node_ids.items():
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

                self.head_results[idx, 0] = head_1 - (hl*(1 - dx/L))
                self.flow_results[idx, 0] = float(self.steady_state_sim.link['flowrate'][pipe])
            else: # Junctions
                self.head_results[idx, 0] = float(self.steady_state_sim.node['head'][node])
                self.flow_results[idx, 0] = float(self.steady_state_sim.node['demand'][node])

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
        self.pipe_ids = {}
        self.valve_ids = {}
                
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
        i = 0; j = 0
        for (n1, n2) in self.network.edges():
            p = self.get_pipe_name(n1, n2)
            link = self.wn.get_link(p)
            if link.link_type == 'Pipe':
                self.pipe_ids[p] = i
                i += 1
            elif link.link_type == 'Valve':
                self.valve_ids[p] = j
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