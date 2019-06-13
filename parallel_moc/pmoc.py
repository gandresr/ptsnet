import wntr
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import subprocess
import dis

from constants import *
from numba import njit
from time import time
from os.path import isdir

# Parallel does not perform well in sandy-bridge architectures
@njit(parallel = False)
def run_interior_step(Q1, Q2, H1, H2, B, R):
    # Keep in mind that the first and last nodes in mesh.nodes will
    #   always be a boundary node
    for i in range(1, len(H1)-1):
        H2[i] = ((H1[i-1] + B[i]*Q1[i-1])*(B[i] + R[i]*abs(Q1[i+1])) \
            + (H1[i+1] - B[i]*Q1[i+1])*(B[i] + R[i]*abs(Q1[i-1]))) \
            / ((B[i] + R[i]*abs(Q1[i-1])) + (B[i] + R[i]*abs(Q1[i+1])))
        Q2[i] = ((H1[i-1] + B[i]*Q1[i-1]) - (H1[i+1] - B[i]*Q1[i+1])) \
            / ((B[i] + R[i]*abs(Q1[i-1])) + (B[i] + R[i]*abs(Q1[i+1])))

@njit
def run_junction_step(
    nodes_up, nodes_down,
    junctions, upstream_num, downstream_num, Q1, Q2, H1, H2, B, R):
    for i in range(junctions.shape[1]):
        sc = 0
        sb = 0
        for j in range(upstream_num[i]):
            n1 = nodes_up[junctions[i,j]]
            n2 = nodes_up[junctions[i,j]]
            sc += (H1[k] + B[k]*Q1[k]) / (B[k] + R[k]*abs(Q1[k]))
            sb += 1 / (B[k] + R[k]*abs(Q1[k]))

        for j in range(upstream_num[i], upstream_num[i]+downstream_num[i]):
            k = junctions[i,j]
            sc += (H1[k] - B[k]*Q1[k]) / (B[k] + R[k]*abs(Q1[k]))
            sb += 1 / (B[k] + R[k]*abs(Q1[k]))

        for j in range(upstream_num[i]):
            k = junctions[i,j]
            H2[j] = sc/sb
            Q2[j] = (H1[k] + B[k]*Q1[k] - H2[k]) / (B[k] + R[k]*abs(Q1[k]))

        for j in range(upstream_num[i], upstream_num[i]+downstream_num[i]):
            k = junctions[i,j]
            H2[j] = sc/sb
            Q2[j] = (H2[j] - H1[j] + B[j]*Q1[j]) / (B[j] + R[j]*abs(Q1[j]))

@njit
def run_reservoir_step(
    reservoir_ids, upstream_nodes, downstream_nodes, Q1, Q2, H1, H2, B, R):
    for i in reservoir_ids:
        H2[i] = H1[i]
        if upstream_nodes[i] == NULL:
            # C- characteristic
            Q2[i] = (H1[i] - H1[downstream_nodes[i]] + B[i]*Q1[downstream_nodes[i]]) \
                / (B[i] + R[i]*abs(Q1[downstream_nodes[i]]))
        elif downstream_nodes[i] == NULL:
            # C+ characteristic
            Q2[i] = (H1[upstream_nodes[i]] + B[i]*Q1[upstream_nodes[i]] - H1[i]) \
                / (B[i] + R[i]*abs(Q1[upstream_nodes[i]]))

def run_valve_step(
    valve_ids, upstream_nodes, downstream_nodes, Q1, Q2, H1, H2, B, R):
    pass

def run_pump_step(
    pump_ids, upstream_nodes, downstream_nodes, Q1, Q2, H1, H2, B, R):
    pass

class Simulation:
    """
    Here all the tables and properties required to
    run a MOC simulation are defined. Tables for
    simulations in parallel are created

    In the meantime:
    * valves are not valid in junctions
    * it is not possible to connect one valve to another
    * valves should be 100% open for initial conditions
    """
    def __init__(self, mesh, T):
        self.mesh = mesh
        self.steady_state_sim = wntr.sim.EpanetSimulator(self.mesh.wn).run_sim()
        self.flow_results = [
            np.zeros(len(mesh.node_name_list), dtype='float64') for i in range(T)]
        self.head_results = [
            np.zeros(len(mesh.node_name_list), dtype='float64') for i in range(T)]
        self.T = T
        self._define_initial_conditions()
        self._define_node_sim_constants()

    def run_simulation(self):
        clk = Clock()

        clk.tic()
        for t in range(1, self.T):
            # HECK YEAH! THIS THING RUNS AT WARP SPEED
            run_interior_step(
                self.flow_results[t-1],
                self.flow_results[t],
                self.head_results[t-1],
                self.head_results[t],
                self.mesh.nodes_float[NODE_FLOAT['B'],:],
                self.mesh.nodes_float[NODE_FLOAT['R'],:])
            # run_junction_step()
            # run_reservoir_step()
            # run_valve_step()
            # run_pump_step()
        clk.toc()

    def _define_node_sim_constants(self):
        # ! ESTIMATE INITIAL CONDITIONS FIRST!!!
        for i, node_name in enumerate(self.mesh.node_name_list):
            link_id = self.mesh.nodes_int[NODE_INT['link_id'], i]

            wave_speed = self.mesh.links_float[LINK_FLOAT['wave_speed'], link_id]
            area = self.mesh.links_float[LINK_FLOAT['area'], link_id]

            ffactor = self.mesh.links_float[LINK_FLOAT['ffactor'], link_id]
            dx = self.mesh.links_float[LINK_FLOAT['dx'], link_id]
            diameter = self.mesh.links_float[LINK_FLOAT['diameter'], link_id]

            # Determine upstream and downstream nodes
            labels = node_name.split('.')
            n1 = labels[0]
            k = int(labels[1])
            n2 = labels[2]
            N = int(labels[4])

            upstream_node = n1 + '.' + str(k - 1) + '.' + '.'.join(labels[2:])
            downstream_node = n1 + '.' + str(k + 1) + '.' + '.'.join(labels[2:])

            if k == N: # end boundary node
                self.mesh.nodes_int[NODE_INT['upstream_node'], i] = self.mesh.node_ids[upstream_node]
                # Notice that downstream_node == NULL
            elif k == 0: # start boundary node
                self.mesh.nodes_int[NODE_INT['downstream_node'], i] = self.mesh.node_ids[downstream_node]
                # Notice that upstream_node == NULL
            else: # interior node
                self.mesh.nodes_int[NODE_INT['upstream_node'], i] = self.mesh.node_ids[upstream_node]
                self.mesh.nodes_int[NODE_INT['downstream_node'], i] = self.mesh.node_ids[downstream_node]

            self.mesh.nodes_float[NODE_FLOAT['B'], i] = wave_speed / (G*area)
            self.mesh.nodes_float[NODE_FLOAT['R'], i] = ffactor*dx / (2*G*diameter*area**2)

    def _define_initial_conditions(self):
        for i, node in enumerate(self.mesh.node_name_list):
            labels = node.split('.')
            n1 = labels[0]
            n2 = labels[2]
            pipe = labels[3]

            k = int(labels[1])
            N = int(labels[4])

            if k == 0 or k == N: # Boundary node
                if k == 0:
                    head = float(self.steady_state_sim.node['head'][n1])
                if k == N:
                    head = float(self.steady_state_sim.node['head'][n2])
                self.flow_results[0][i] = float(self.steady_state_sim.link['flowrate'][pipe])
            else: # Interior node
                head_1 = float(self.steady_state_sim.node['head'][n2])
                head_2 = float(self.steady_state_sim.node['head'][n1])
                hl = head_1 - head_2
                L = self.mesh.wn.get_link(pipe).length
                dx = k * L / self.mesh.segments[pipe]
                self.head_results[0][i] = head_1 - (hl*(1 - dx/L))
                self.flow_results[0][i] = float(self.steady_state_sim.link['flowrate'][pipe])

class Mesh:
    """ Defines the mesh for an EPANET network to solve the 1D MOC

    This class allows the creation of a Mesh object to
    solve the method of characteristics. The mesh is created
    based on an EPANET .inp file with information of the
    WDS network. Information is extracted from the .inp
    using WNTR and networkx.

    In order to create the mesh, it is necessary that the user
    defines wave speed values for every pipe in the network. This
    can be done through a file or by specifying a default wave speed
    value for all the pipes in the network.

    The mesh is created based on the CFL condition and
    wave speed values are adjusted to satisfy that all the
    pipes have the same time step value in the space-time grid

    Important considerations:
        # * The network graph is generated by the WNTR library
        # * The Mesh is a segmented network that includes
            new nodes between pipes which are denominated interior points
        # * The nodes in the network graph are denominated junctions
        # * Indexes are considered ids and object names are EPANET identifiers
        # * MOC nodes are labeled as follows: 'initial_node.k.end_node.link_name.segments_num'
        # * Two junctions cannot be connected by a non-pipe element
        # * IDS are indexes associated to data structures
        # * Junctions in EPANET are not the same as junctions in PMOC
    """
    def __init__(self, input_file, dt, wave_speed_file = None, default_wave_speed = None):
        """Creates a Mesh object from a .inp EPANET file

        The MOC Mesh is created based on a desired time step which is the
        the same for every pipe in the network. This is achieved by adjusting
        the wave speed values of the pipes

        Arguments:
            input_file {string} -- path to EPANET's .inp file of the water network
            dt {float} -- desired time step for the MOC simulation

        Keyword Arguments:
            wave_speed_file {string} -- path to the file that contains information
                of the wave speed values for the pipes in the network (default: {None})
            default_wave_speed {float} -- wave speed value for all the pipes in
                the network (default: {None})

        Raises:
            Exception: If no definition of default_wave_speed value or wave_speed_file
                is given. At least one should be given
        """

        self.fname = input_file[:input_file.find('.inp')]
        self.wn = wntr.network.WaterNetworkModel(input_file)
        self.network_graph = self.wn.get_graph()
        self.mesh_graph = None
        self.time_step = None
        self.num_processors = None # number of processors

        # Number of segments are only defined for pipes
        self.segments = None

        # Wavespeed values associated to each link
        self.wave_speeds = None

        # The MOC-mesh_graph graph can be traversed from a root node,
        # i.e., a node whose degree is equal to 1
        self.root_node = None

        # Data structures associated to nodes (only boundary nodes and interior nodes)
        self.nodes_int = None
        self.nodes_float = None
        self.node_ids = None
        self.boundary_ids = None
        self.node_name_list = None

        # Data structures associated to junctions
        self.junctions = None
        self.junction_ids = None
        self.junction_name_list = None

        # Data structures associated to links (pipes, valves, pumps)
        self.links_int = None
        self.links_float = None
        self.link_ids = None
        self.link_name_list = None

        # Initialize mesh
        if (default_wave_speed is None) and (wave_speed_file is None):
            raise Exception("It is necessary to define a wave speed value or waves speed file")
        self.initialize(dt, wave_speed_file, default_wave_speed)

        # self.partitioning: Contains graph partitioning info
        #   to distribute work among processors
        #
        #   When created, it is a dictionary whose keys
        #   are the names of the nodes in the mesh_graph
        #   and the values are pairs (p, s) where p is
        #   the processor to which the node belongs, and
        #   s indicates if a node is a separator according
        #   to parHIP
        self.partitioning = None

    def _define_wave_speeds(self, default_wave_speed = None, wave_speed_file = None):
        """ Stores the values of the wave speeds for every pipe in the EPANET network

        The file should be a CSV file, specifying the pipe and its wave_speed as follows:
            pipe_name_1,wave_speed_1
            pipe_name_2,wave_speed_n
            ...
            pipe_name_n,wave_speed_n

        Pipes not specified in the file will have a wave_speed value
        equal to the default_wave_speed

        Keyword Arguments:
            default_wave_speed {float} -- default value of wave speed for all the pipes in
            the network (default: {None})
            wave_speed_file {string} -- path to CSV file that contains information of the wave speeds
            for each pipe in the network (default: {None})

        Raises:
            Exception: If default_wave_speed is not defined and the file with information
            of the wave speeds is incomplete
        """
        # TODO : TEST IF TIME STEPS ARE THE SAME FOR ALL THE SEGMENTS/PIPES(?)
        self.wave_speeds = {}

        if default_wave_speed is not None:
            self.wave_speeds = dict.fromkeys(self.wn.pipe_name_list, default_wave_speed)

        if wave_speed_file:
            with open(wave_speed_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    pipe, wave_speed = line.split(',')
                    self.wave_speeds[pipe] = float(wave_speed)

        if len(self.wave_speeds) != len(self.wn.pipe_name_list):
            self.wave_speeds = {}
            raise Exception("""
            The file does not specify wave speed values for all the pipes,
            it is necessary to define a default wave speed value""")

    def _define_segments(self, dt):
        """Estimates the number of segments for each pipe in the EPANET network

        Pipes are segmented in order to create a Mesh for the MOC

        Arguments:
            dt {float} -- desired time step
        """
        # Get the maximum time steps for each pipe
        self.segments = self.wn.query_link_attribute('length') # The length attribute is just for pipes

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

        # It is necessary to redefine the mesh graph everytime the segments are redefined
        self._define_mesh()

    def _define_mesh(self, write_mesh = False):
        """Defines the mesh graph of the water network

        The physical network given by WNTR as a networkx Graph
        is segmented to create a mesh for the MOC. Then, a
        mesh graph is created based on the network graph
        and interior nodes and junction boundaries are created
        and labeled using the following convention:

        'initial_node.k.end_node.link_name.segments_num'

        where:

        initial_node: name of the start node of the pipe to which
            the interior point belongs
        k: interior point order in the pipe
        end_node: name of the end node of the pipe to which
            the interior point belongs
        link_name: name of the pipe to which the interior
            point belongs
        segments_num: total number of segments in pipe

        # * There are sp-1 interior points for every pipe (sp is the
            number of segments in which a pipe is going to be divided
        # * Every pipe has 2 junction boundary nodes
        # * This function should be called only after defining the segments
            for each pipe in the network
        """

        # The segmented MOC-mesh_graph graph is generated
        self.mesh_graph = nx.Graph()

        # parfor
        # nb : Node at the beginning of the edge
        # ne : Node at the end of the edge
        for i, nb in enumerate(self.network_graph):
            # A root node is chosen
            if self.root_node is None:
                if self.network_graph.degree(nb) == 1:
                    self.root_node = nb

            for neighbor in self.network_graph[nb]:
                for p in self.network_graph[nb][neighbor]:
                    n1 = nb
                    link = self.wn.get_link(p)
                    if link.link_type == 'Pipe':
                        # interior points and boundary nodes are created (ni)
                        for j in range(self.segments[p]+1): # (s-1) interior points, (2) boundary nodes
                            # 'initial_node.k.end_node.link_name.segments_num'
                            # * nodes with k == 0 or k == self.segments[p] are junction boundaries
                            ni = nb + '.' + str(j) + '.' + neighbor + '.' + p + '.' + str(self.segments[p])
                            self.mesh_graph.add_edge(n1, ni)
                            n1 = ni
                    self.mesh_graph.add_edge(n1, neighbor)

        if write_mesh:
            self._write_mesh()

    def _write_mesh(self):
        """ Saves the mesh graph in a file compatible with METIS

        # * This function should only be called after defining the mesh_graph

        Raises:
            Exception: if mesh graph has not been defined
        """
        G = self.mesh_graph
        ids = dict( zip( list(G), range(len(G)) ) )

        # Network is stored in METIS format
        if G:
            with open(self.fname + '.graph', 'w') as f:
                f.write("%d %d\n" % (len(G), len(G.edges())))
                for node in ids:
                    fline = "" # file content
                    for neighbor in G[node]:
                        fline += "%d " % (ids[neighbor] + 1)
                    fline += '\n'
                    f.write(fline)
        else:
            raise Exception("It is necessary to define the mesh graph")

    def _define_links(self):
        """Defines data structures associated to links

        There are three data structures associated to links:
        - self.links: table with properties of links (as defined in LINK)
        - self.link_name_list: list with the names of links. The order directly
            corresponds to the order of self.links
        - self.link_ids: dict that maps link names to indexes associated to
            data structures
        """

        self.link_ids = {}
        self.link_name_list = []

        i = 0
        self.links_int = np.full((len(LINK_INT), self.wn.num_links), NULL, dtype='int')
        self.links_float = np.full((len(LINK_FLOAT), self.wn.num_links), NULL, dtype='float64')

        for link_name in self.wn.links:
            link = self.wn.get_link(link_name)

            if link.link_type == 'Pipe':
                self.links_float[LINK_FLOAT['wave_speed'], i] = self.wave_speeds[link_name]
                self.links_float[LINK_FLOAT['length'], i] = link.length
                self.links_float[LINK_FLOAT['dx'], i] = link.length / self.segments[link_name]

            # The setting id and Pipes friction factor are set in the Simulation class
            self.links_int[LINK_INT['id'], i] = i
            self.links_int[LINK_INT['link_type'], i] = LINK_TYPES[link.link_type]
            self.links_float[LINK_FLOAT['diameter'], i] = link.diameter
            self.links_float[LINK_FLOAT['area'], i] = np.pi * link.diameter**2 /4
            self.link_name_list.append(link_name)
            self.link_ids[link_name] = i
            i += 1

    def _define_nodes(self):
        """Defines data structures associated to nodes

        There are three data structures associated to nodes:
        - self.nodes: table with properties of nodes (as defined in NODE)
        - self.node_name_list: list with the names of nodes. The order directly
            corresponds to the order of self.nodes
        - self.node_ids: dict that maps node names to indexes associated to
            data structures

        Raises:
            Exception: if there is a junction that has only non-pipe elements
                attached to it
            Exception: if there is an isolated node
        """


        # Total nodes in the analysis:
        #  boundary nodes + interior nodes
        total_nodes_num = len(self.mesh_graph) - len(self.network_graph)

        self.nodes_int = np.full((len(NODE_INT), total_nodes_num), NULL, dtype='int')
        self.nodes_float = np.full((len(NODE_FLOAT), total_nodes_num), NULL, dtype='float64')

        self.node_ids = {}
        self.boundary_ids = []
        self.node_name_list = []

        # Node information is stored in predorder
        #   in order to preserve memory locality
        dfs = nx.dfs_preorder_nodes(
            self.mesh_graph,
            source = self.root_node) # Boundary node

        i = 0
        for node in dfs:
            # Remember that mesh_graph is an undirected networkx Graph
            if '.' in node:

                # Store node id
                self.node_ids[node] = i

                labels = node.split('.')
                link_name = labels[3]
                k = int(labels[1]) # internal index of node in pipe
                N = int(labels[4]) # Total number of segments in pipe

                self.nodes_int[NODE_INT['id'], i] = i
                self.nodes_int[NODE_INT['link_id'], i] = self.link_ids[link_name]

                if k == 0 or k == N: # Boundary nodes
                    neighbors = list(self.mesh_graph.neighbors(node)) # len(neighbors) <= 2
                    for neighbor in neighbors:
                        if '.' not in neighbor:
                            neighbor_links = self.wn.get_links_for_node(neighbor)
                            # Check if boundary node is reservoir
                            if neighbor in self.wn.reservoir_name_list:
                                self.nodes_int[NODE_INT['node_type'], i] = NODE_TYPES['reservoir']
                                self.boundary_ids.append(i)
                            # TODO (TEST) ALL BOUNDARY NODES SHOULD HAVE AT MOST TWO NEIGHBORS AND ONE OF THEM
                            #   HAS TO BE A JUNCTION. I.E., NO '.' IN NEIGHBOR NAME
                            # Check if boundary node is junction
                            elif len(neighbor_links) > 2:
                                non_pipes_count = 0
                                for n_link_name in neighbor_links:
                                    n_link = self.wn.get_link(n_link_name)
                                    if n_link.link_type in ('Valve', 'Pump'): # WNTR link types
                                        non_pipes_count += 1
                                if non_pipes_count == len(neighbor_links):
                                    # TODO (TEST) IF A NODE'S NEIGHBOR HAS MORE THAN TWO NEIGHBORS
                                    #   THE NODE'S NEIGHBOR HAS TO HAVE AT LEAST ONE PIPE ELEMENT
                                    raise Exception('Only non-pipe elements connected to %s' % neighbor)
                                self.nodes_int[NODE_INT['node_type'], i] = NODE_TYPES['junction']
                                self.boundary_ids.append(i)
                            elif len(neighbor_links) > 0:
                                non_pipes_count = 0
                                for n_link_name in neighbor_links:
                                    n_link = self.wn.get_link(n_link_name)
                                    # Check if boundary node is valve
                                    if n_link.link_type == 'Valve':
                                        if n_link.start_node_name == neighbor:
                                            self.nodes_int[NODE_INT['node_type'], i] = NODE_TYPES['valve_start']
                                            self.boundary_ids.append(i)
                                        elif n_link.end_node_name == neighbor:
                                            self.nodes_int[NODE_INT['node_type'], i] = NODE_TYPES['valve_end']
                                            self.boundary_ids.append(i)
                                        self.nodes_int[NODE_INT['link_id'], i] = self.link_ids[n_link_name]
                                        non_pipes_count += 1
                                    # Check if boundary node is pump
                                    elif n_link.link_type == 'Pump':
                                        if n_link.start_node_name == neighbor:
                                            self.nodes_int[NODE_INT['node_type'], i] = NODE_TYPES['pump_start']
                                            self.boundary_ids.append(i)
                                        elif n_link.end_node_name == neighbor:
                                            self.nodes_int[NODE_INT['node_type'], i] = NODE_TYPES['pump_end']
                                            self.boundary_ids.append(i)
                                        self.nodes_int[NODE_INT['link_id'], i] = self.link_ids[n_link_name]
                                        non_pipes_count += 1
                                if non_pipes_count == 2:
                                    raise Exception('Only non-pipe elements connected to %s' % neighbor)
                                elif non_pipes_count == 0:
                                    self.nodes_int[NODE_INT['node_type'], i] = NODE_TYPES['junction']
                                    self.boundary_ids.append(i)
                            else:
                                raise Exception('%s is an isolated node' % neighbor)
                else:
                    self.nodes_int[NODE_INT['node_type'], i] = NODE_TYPES['interior']
                self.node_name_list.append(node)
                self.node_ids[node] = i
                i += 1

    def _define_junctions(self):
        """Defines data structures associated to junctions

        There are three data structures associated to junctions:
        - self.junctions: table with properties of junctions (as defined in JUNCTION)
        - self.junction_name_list: list with the names of junctions. The order directly
            corresponds to the order of self.junctions
        - self.junction_ids: dict that maps junction names to indexes associated to
            data structures

        Raises:
            Exception: if two junctions are connected by a non-pipe element
            Exception: if a junction has more than MAX_NEIGHBORS_IN_JUNCTION links attached
                to it
        """

        self.junctions = np.full((len(JUNCTION_INT), len(self.network_graph)), NULL, dtype = int)
        self.junction_ids = {}
        self.junction_name_list = []

        i = 0
        for junction in self.network_graph:
            neighbors = self.mesh_graph.neighbors(junction)
            upstream = []
            downstream = []

            # The neighbors of a junction node can be:
            #   - Start/end nodes of a non-pipe element
            #   - Boundary nodes of a pipe
            for neighbor in neighbors:
                if '.' in neighbor: # Boundary nodes of a pipe
                    labels = neighbor.split('.')
                    k = int(labels[1])
                    N = int(labels[4])
                    if k == 0:
                        downstream.append(neighbor)
                    elif k == N:
                        upstream.append(neighbor)
                else: # Start/end nodes of a non-pipe element
                    second_neighbors = list(set(self.mesh_graph.neighbors(neighbor)) - {junction})
                    # Since two junctions cannot be connected by non-pipe elements,
                    #   the neighbors of an start/end node, exluding the junction
                    #   node connected to it, can only be boundary nodes of a pipe. Thus,
                    #   len(second_neighbors) \in {0,1}
                    if len(second_neighbors) == 0: # Terminal junction
                        continue
                    if '.' not in second_neighbors[0]:
                        raise Exception('Internal error, assumptions are not satisfied')
                    labels = second_neighbors[0].split('.')
                    k = int(labels[1])
                    N = int(labels[4])
                    if k == N:
                        upstream.append(second_neighbors[0])
                    else:
                        downstream.append(second_neighbors[0])

            if len(upstream) + len(downstream) > MAX_NEIGHBORS_IN_JUNCTION:
                raise Exception('Junction %s has too many links (max %d)' % (junction, MAX_NEIGHBORS_IN_JUNCTION))

            self.junctions[JUNCTION_INT['upstream_neighbors_num'], i] = len(upstream)
            self.junctions[JUNCTION_INT['downstream_neighbors_num'], i] = len(downstream)

            j = 1
            for node_name in upstream:
                self.junctions[JUNCTION_INT['n%d' % j], i] = self.node_ids[node_name]
                self.junctions[JUNCTION_INT['p%d' % j], i] = self.get_processor(node_name)
                j += 1
            for node_name in downstream:
                self.junctions[JUNCTION_INT['n%d' % j], i] = self.node_ids[node_name]
                self.junctions[JUNCTION_INT['p%d' % j], i] = self.get_processor(node_name)
                j += 1

            self.junction_name_list.append(junction)
            self.junction_ids[junction] = i
            i += 1

    def initialize(self, dt, wave_speed_file, default_wave_speed):
        """Initializes the Mesh object

        Arguments:
            dt {float} -- desired time step for the MOC simulation
            wave_speed_file {string} -- path to the file that contains information
                of the wave speed values for the pipes in the network
            default_wave_speed {float} -- wave speed value for all the pipes in
                the network
        """
        self._define_wave_speeds(default_wave_speed, wave_speed_file)
        self._define_segments(dt)
        self._define_mesh()
        self._define_links()
        self._define_nodes()
        self._define_junctions()
        self.partitioning = None

    def define_partitions(self, k):
        """Defines network partitioning using parHIP (external lib)

        Arguments:
            k {integer} -- desired number of partitions
        """

        if not isdir(MOC_PATH + 'partitionings'):
            subprocess.call(['mkdir', MOC_PATH + 'partitionings'])

        self._write_mesh()
        script = MOC_PATH + 'parHIP/kaffpa'
        subprocess.call([
            script, self.fname + '.graph',
            '--k=' + str(k),
            '--preconfiguration=strong',
            '--output_filename=' + MOC_PATH + 'partitionings/p%d.graph' % k],
            stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)

        script = MOC_PATH + 'parHIP/partition_to_vertex_separator'

        subprocess.call([
            script, self.fname + '.graph',
            '--k=' + str(k),
            '--input_partition=' + MOC_PATH + 'partitionings/p%d.graph' % k,
            '--output_filename=' + MOC_PATH + 'partitionings/s%d.graph' % k],
            stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)

        self.num_processors = k
        pp = zip(
            np.loadtxt(MOC_PATH + 'partitionings/p%d.graph' % k, dtype=int),
            np.loadtxt(MOC_PATH + 'partitionings/s%d.graph' % k, dtype=int))

        # Stored in the same order of mesh_graph:
        names = list(self.mesh_graph)
        self.partitioning = dict(zip(names, pp))

        for i, node in enumerate(self.node_name_list):
            self.nodes_int[NODE_INT['processor'], i] = self.partitioning[node][0] # processor
            # is separator? ... more details in parHIP user manual
            self.nodes_int[NODE_INT['is_ghost'], i] = self.partitioning[node][1] == self.num_processors

    def get_processor(self, node):
        """Returns the processor assigned to a node in the mesh graph

        Arguments:
            node {string} -- name of the node

        Returns:
            integer -- processor id
        """
        return self.nodes_int[NODE_INT['processor'], self.node_ids[node]]

class Clock:
    """Wall-clock time
    """
    def __init__(self):
        self.clk = time()

    def tic(self):
        """Starts timer
        """
        self.clk = time()

    def toc(self):
        """Ends timer and prints time elapsed
        """
        print('Elapsed time: %.8f seconds' % (time() - self.clk))