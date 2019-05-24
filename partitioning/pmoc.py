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

class Simulation:
    # Enumerators

    class Node(enum.Enum):
        node_type = 0 # {none, reservoir, junction, end, valve_a, valve_b}
        link_id = 1 # {none (-1), link_id, valve_id}
        neighbors_id = 2
        processor = 3
        is_ghost = 4
  
    class Pipe(enum.Enum):
        node_a = 0
        node_b = 1
        diameter = 2
        area = 3
        wave_speed = 4
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
        link_a = 4
        link_b = 5 

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
        Requires an Mesh
        T: total time steps
        '''
        self.mesh = network
        self.time_steps = T

        # Simulation results
        self.steady_state_sim = wntr.sim.EpanetSimulator(network.wn).run_sim()
        self.flow_results = np.zeros( (len(network.mesh_graph), T) )
        self.head_results = np.zeros( (len(network.mesh_graph), T) )
        self.upstream_flow_results = []
        self.downstream_flow_results = []
        
        # a[a[:,self.Node.processor.value].argsort()] - Sort by processor
        self.nodes = np.zeros((len(network.mesh_graph), len(self.Node)), dtype=int)
        self.upstream_pipes = []
        self.downstream_pipes = []
        self.upstream_nodes = []
        self.downstream_nodes = []

        for i in range(len(network.mesh_graph)):
            self.upstream_pipes.append( [] )
            self.downstream_pipes.append( [] )
            self.upstream_nodes.append( [] )
            self.downstream_nodes.append( [] )
        
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
            node_name = self.mesh.node_name_list[i]
            u_pipes = self.upstream_pipes[i]
            d_pipes = self.downstream_pipes[i]
            u_nodes = self.upstream_nodes[i]
            d_nodes = self.downstream_nodes[i]

            if node_type == self.node_types.reservoir.value:
                pass
            elif node_type == self.node_types.interior.value:
                if len(u_pipes) != 1 or len(d_pipes) != 1:
                    raise Exception("There is an error with the data structures")

                u_node_id = self.mesh.node_ids[u_nodes[0]]
                d_node_id = self.mesh.node_ids[d_nodes[0]]
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

                wave_speed = self.pipes[pipe_id, self.Pipe.wave_speed.value]
                area = self.pipes[pipe_id, self.Pipe.area.value]
                ffactor = self.pipes[pipe_id, self.Pipe.ffactor.value]
                diameter = self.pipes[pipe_id, self.Pipe.diameter.value]
                dx = self.pipes[pipe_id, self.Pipe.dx.value]

                B = wave_speed/(g*area)
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

                    u_pipe_id = self.mesh.link_ids[u_pipe]
                    u_node_id = self.mesh.node_ids[u_node]

                    wave_speed = self.pipes[u_pipe_id, self.Pipe.wave_speed.value]
                    area = self.pipes[u_pipe_id, self.Pipe.area.value]
                    ffactor = self.pipes[u_pipe_id, self.Pipe.ffactor.value]
                    diameter = self.pipes[u_pipe_id, self.Pipe.diameter.value]
                    dx = self.pipes[u_pipe_id, self.Pipe.dx.value]

                    B = wave_speed/(g*area)
                    R = ffactor*dx/(2*g*diameter*area**2)
                    H1 = self.head_results[u_node_id, t-1]
                    Q1 = self.flow_results[u_node_id, t-1]

                    Cp[j] = H1 + B*Q1
                    Bp[j] = B + R*abs(Q1)
                    sc += Cp[j]/Bp[j]
                    sb += 1/Bp[j]

                for j, d_pipe in enumerate(d_pipes):
                    d_node = d_nodes[j]

                    d_pipe_id = self.mesh.link_ids[d_pipe]
                    d_node_id = self.mesh.node_ids[d_node]

                    wave_speed = self.pipes[d_pipe_id, self.Pipe.wave_speed.value]
                    area = self.pipes[d_pipe_id, self.Pipe.area.value]
                    ffactor = self.pipes[d_pipe_id, self.Pipe.ffactor.value]
                    diameter = self.pipes[d_pipe_id, self.Pipe.diameter.value]
                    dx = self.pipes[d_pipe_id, self.Pipe.dx.value]

                    B = wave_speed/(g*area)
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

                node_name = self.mesh.node_name_list[i]
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
        for node, node_id in self.mesh.node_ids.items():

            ## TYPE & LINK_ID ARE DEFINED
            # ----------------------------------------------------------------------------------------------------------------

            # Remember that mesh_graph is an undirected networkx Graph
            neighbors = list(self.mesh.mesh_graph.neighbors(node))
            if node in self.mesh.wn.reservoir_name_list:
                # Check if node is reservoir node
                self.nodes[node_id, self.Node.node_type.value] = self.node_types.reservoir.value
                self.nodes[node_id, self.Node.link_id.value] = -1
            # Check if the node belongs to a valve
            if self.nodes[node_id, self.Node.node_type.value] == self.node_types.none.value: # Type not defined yet
                if not '.' in node:
                    for valve, valve_id in self.mesh.valve_ids.items():
                        link = self.mesh.wn.get_link(valve)
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
                        self.nodes[node_id, self.Node.link_id.value] = self.mesh.link_ids[pipe]
                    else:
                        self.nodes[node_id, self.Node.node_type.value] = self.node_types.junction.value
                        self.upstream_flow_results.append( [] )
                        self.downstream_flow_results.append( [] )
                        self.nodes[node_id, self.Node.link_id.value] = -1

            # ----------------------------------------------------------------------------------------------------------------

            ## PROCESSOR & IS_GHOST ARE DEFINED
            # ----------------------------------------------------------------------------------------------------------------

            self.nodes[node_id, self.Node.processor.value] = self.mesh.get_processor(node)
            self.nodes[node_id, self.Node.is_ghost.value] = (self.mesh.separator[node_id] == self.mesh.num_processors)

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
                        pipe = self.mesh.get_pipe_name(n, node)
                        if pipe == None:
                            pipe = self.mesh.get_pipe_name(node, n)
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
        for pipe, pipe_id in self.mesh.link_ids.items():
            link = self.mesh.wn.get_link(pipe)
            self.pipes[pipe_id, self.Pipe.node_a.value] = self.mesh.node_ids[link.start_node_name]
            self.pipes[pipe_id, self.Pipe.node_b.value] = self.mesh.node_ids[link.end_node_name]
            diameter = link.diameter
            self.pipes[pipe_id, self.Pipe.diameter.value] = diameter
            self.pipes[pipe_id, self.Pipe.area.value] = np.pi*diameter**2/4
            self.pipes[pipe_id, self.Pipe.wave_speed.value] = self.mesh.wave_speeds[pipe]
            self.pipes[pipe_id, self.Pipe.ffactor.value] = float(self.steady_state_sim.link['frictionfact'][pipe])
            self.pipes[pipe_id, self.Pipe.length.value] = link.length
            self.pipes[pipe_id, self.Pipe.dx.value] = link.length / self.mesh.segments[pipe]

    def _define_valves(self):
        for valve, valve_id in self.mesh.valve_ids.items():
            link = self.mesh.wn.get_link(valve)
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
        valve_id = self.mesh.valve_ids[valve_name]

        T = min(len(settings), self.time_steps)
            
        for t in range(T):
            self.valve_settings[valve_id, t] = settings[t]

    def _define_initial_conditions(self):
        for node, node_id in self.mesh.node_ids.items():
            if '.' in node: # interior points
                labels = node.split('.') # [n1, k, n2, p]
                n1 = labels[0]
                k = abs(int(labels[1]))
                n2 = labels[2]
                pipe = labels[3]
                
                head_1 = float(self.steady_state_sim.node['head'][n2])
                head_2 = float(self.steady_state_sim.node['head'][n1])
                hl = head_1 - head_2
                L = self.mesh.wn.get_link(pipe).length
                dx = k * L / self.mesh.segments[pipe]
                
                self.head_results[node_id, 0] = head_1 - (hl*(1 - dx/L))
                self.flow_results[node_id, 0] = float(self.steady_state_sim.link['flowrate'][pipe])
            else:
                head = float(self.steady_state_sim.node['head'][node])
                for j, neighbor in enumerate(self.upstream_nodes[node_id]):
                    link_name = None
                    neighbor_id = self.mesh.node_ids[neighbor]
                    if self.nodes[neighbor_id, self.Node.node_type.value] in (self.node_types.valve_a, self.node_types.valve_b):
                        link_name = self.mesh.valve_names[self.nodes[neighbor_id, self.Node.link_id.value]]
                    else:
                        idx = int(self.nodes[neighbor_id, self.Node.link_id.value])
                        link_name = self.mesh.link_name_list[idx]
                    
                    if len(list(self.mesh.mesh_graph.neighbors(node))) > 2:
                        junction_id = self.junction_ids[node]
                        self.upstream_flow_results[junction_id][j][0] = float(
                            self.steady_state_sim.link['flowrate'][link_name])
                for j, neighbor in enumerate(self.downstream_nodes[node_id]):
                    link_name = None
                    neighbor_id = self.mesh.node_ids[neighbor]
                    if self.nodes[neighbor_id, self.Node.node_type.value] in (self.node_types.valve_a, self.node_types.valve_b):
                        link_name = self.mesh.valve_names[self.nodes[neighbor_id, self.Node.link_id.value]]
                    else:
                        link_name = self.mesh.link_name_list[int(self.nodes[neighbor_id, self.Node.link_id.value])]
                    
                    if len(list(self.mesh.mesh_graph.neighbors(node))) > 2:
                        junction_id = self.junction_ids[node]
                        self.downstream_flow_results[junction_id][j][0] = float(
                            self.steady_state_sim.link['flowrate'][link_name])

                self.head_results[node_id, 0] = head

class Mesh:
    '''
    This class allows the creation of a Mesh object to 
    solve the method of characteristics. The mesh is created
    based on an EPANET .inp file with information of the 
    WDS network. Information is extracted from the .inp 
    using WNTR and networkx. 
    
    In order to create the mesh, it is necessary that the user
    defines the wave_speeds for every pipe in the network. This
    can be done through a file or by specifying a default_wave_speed
    for all the pipes in the network.

    The mesh is created based on the CFL condition and 
    wave speed values are adjusted to satisfy that all the 
    pipes have the same time step value in the space-time grid    
    '''
    def __init__(self, 
        input_file, 
        dt, 
        wave_speed_file = None, 
        default_wave_speed = 1200, 
        file_separator = ','):

        '''

        * The network graph is generated by WNTR
        * The Mesh is a segmented network that includes 
            new nodes between pipes which are denominated interior points
        * The nodes in the network graph are denominated junctions
        '''
        self.fname = input_file[:input_file.find('.inp')]
        self.wn = wntr.network.WaterNetworkModel(input_file)
        self.sim_graph = self.wn.get_graph()
        self.mesh_graph = None
        self.time_step = None
        self.num_processors = None # number of processors

        # Number of segments are only defined for pipes
        self.segments = None
        
        # Ids for nodes, pipes, and valves
        self.node_ids = {}
        self.node_name_list = []
        
        self.link_ids = {}
        self.link_name_list = []
        
        self.wave_speeds = {}

        # Create mesh
        self._define_wave_speeds(default_wave_speed, wave_speed_file)
        self._define_segments(dt)
        self._define_mesh()
        self._define_ids()

        # Graph partitioning for distributed-memory
        self.partition = None
        self.separator = None

    def _define_wave_speeds(self, default_wave_speed = None, wave_speed_file = None):
        '''
        Stores the values of the wave speeds for every pipe in the
        EPANET network.

        Inputs: 
        - default_wave_speed: 
        - wave_speed_file:

        The file should be a CSV file, specifying the pipe and its 
        wave_speed as follows:

            pipe_name_1,wave_speed_1
            pipe_name_2,wave_speed_n
            ...
            pipe_name_n,wave_speed_n

        Pipes not specified in the file will have a wave_speed value
        equal to the default_wave_speed
        '''
        
        if default_wave_speed is not None:
            self.wave_speeds = dict.fromkeys(self.wn.pipe_name_list, default_wave_speed)

        if wave_speed_file:
            with open(wave_speed_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    pipe, wave_speed = line.split(',')
                    self.wave_speeds[pipe] = float(wave_speed)

    def _define_segments(self, dt):
        # Get the maximum time steps for each pipe
        self.segments = self.wn.query_link_attribute('length') # The lenght attribute is just for pipes

        for pipe in self.segments:
            self.segments[pipe] /= self.wave_speeds[pipe]
        
        # Maximum dt in the system to capture waves in all pipes
        max_dt = self.segments[min(self.segments, key=self.segments.get)]

        # Desired dt < max_dt ?
        t_step = min(dt, max_dt)
        self.time_step = t_step

        # The number of segments is defined
        for pipe in self.segments:
            self.segments[pipe] /= t_step
            # The wave_speed is adjusted to compensate the truncation error
            e = int(self.segments[pipe])-self.segments[pipe] # truncation error
            self.wave_speeds[pipe] = self.wave_speeds[pipe]/(1 + e/self.segments[pipe])
            self.segments[pipe] = int(self.segments[pipe])

    def _define_mesh(self):
        '''
        This function should be called only after defining the segments
        for each pipe in the network
        '''

        G = self.sim_graph
        
        # The segmented MOC-mesh_graph graph is generated
        self.mesh_graph = nx.Graph() 
        
        # The MOC-mesh_graph graph will be traversed from a boundary node
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
                        self.mesh_graph.add_edge(n1, ni)
                        n1 = ni

                    self.mesh_graph.add_edge(n1, neighbor)
        
        # parfor
        self._define_ids()

    def _define_ids(self):
        for i, node in enumerate(self.mesh_graph):
            self.node_ids[node] = i
            self.node_name_list.append(node)
        i = 0
        for (n1, n2) in self.sim_graph.edges():
            link_name = self.get_link_name(n1, n2)
            link = self.wn.get_link(link_name)
            self.link_ids[link_name] = i
            self.link_name_list.append(link_name)
            i += 1

    def _write_mesh(self):
        '''
        This function should only be called after defining the mesh_graph
        '''
        G = self.mesh_graph
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
        self._write_mesh()
        script = './parHIP/kaffpa'
        subprocess.call([
            script, self.fname + '.graph', 
            '--k=' + str(k), 
            '--preconfiguration=strong', 
            '--output_filename=partitionings/p%d.graph' % k])

        if k == 2:
            script = './parHIP/node_separator'
        else:
            script = './parHIP/partition_to_vertex_separator'

        subprocess.call([
            script, self.fname + '.graph', 
            '--k=' + str(k), 
            '--input_partition=partitionings/p%d.graph' % k, 
            '--output_filename=partitionings/s%d.graph' % k])

        self.num_processors = k
        self.partition = np.loadtxt('partitionings/p%d.graph' % k, dtype=int)
        self.separator = np.loadtxt('partitionings/s%d.graph' % k, dtype=int)
        
    def get_processor(self, node):
        return self.partition[self.node_ids[node]]
    
    def get_link_name(self, n1, n2):
        try:
            for p in self.sim_graph[n1][n2]:
                return p
        except:
            return None

class Clock:
    def __init__(self):
        self.clk = time()
    
    def tic(self):
        self.clk = time()
    
    def toc(self):
        print('Elapsed time: %f seconds' % (time() - self.clk))