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
def run_junction_step(junctions, Q1, Q2, H1, H2, B, R):
    # junctions is a matrix that contains neighbors info for each boundary node
    #   bare in mind that neighbors of boundary nodes are also boundary nodes!
    for i in range(junctions.shape[1]):
        sc = 0
        sb = 0
        for j in range(junctions[0,i]): # junctions[0,i] == upstream_neigh_num[i]
            k = junctions[j+2,i]-1
            sc += (H1[k] + B[k]*Q1[k]) / (B[k] + R[k]*abs(Q1[k]))
            sb += 1 / (B[k] + R[k]*abs(Q1[k]))

        for j in range(junctions[0,i], junctions[0,i]+junctions[1,i]):
            k = junctions[j+2,i]+1
            sc += (H1[k] - B[k]*Q1[k]) / (B[k] + R[k]*abs(Q1[k]))
            sb += 1 / (B[k] + R[k]*abs(Q1[k]))

        for j in range(junctions[0,i]):
            k = junctions[j+2,i]-1
            H2[j] = sc/sb
            Q2[j] = (H1[k] + B[k]*Q1[k] - H2[k]) / (B[k] + R[k]*abs(Q1[k]))

        for j in range(junctions[0,i], junctions[0,i]+junctions[1,i]):
            k = junctions[j+2,i]+1
            H2[j] = sc/sb
            Q2[j] = (H2[k] - H1[k] + B[k]*Q1[k]) / (B[k] + R[k]*abs(Q1[k]))

@njit
def run_reservoir_step(
    reservoir_ids, is_start, Q1, Q2, H1, H2, B, R):
    for i in reservoir_ids:
        H2[i] = H1[i]
        if is_start[i] == True: # Start boundary node
            # C- characteristic
            Q2[i] = (H1[i] - H1[i+1] + B[i-1]*Q1[i+1]) \
                / (B[i-1] + R[i-1]*abs(Q1[i+1]))
        elif is_start[i] == False: # End boundary node
            # C+ characteristic
            Q2[i] = (H1[i-1] + B[i-1]*Q1[i-1] - H1[i]) \
                / (B[i-1] + R[i-1]*abs(Q1[i-1]))

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
        self.T = T
        self.mesh = mesh
        self.steady_state_sim = wntr.sim.EpanetSimulator(self.mesh.wn).run_sim()
        self.flow_results = np.zeros((T, len(mesh.node_name_list)), dtype='float64')
        self.head_results = np.zeros((T, len(mesh.node_name_list)), dtype='float64')
        self._define_initial_conditions()
        self._define_node_sim_constants()

    def run_simulation(self):
        clk = Clock()

        clk.tic()
        for t in range(1, self.T):
            # HECK YEAH! THIS THING RUNS AT WARP SPEED
            run_interior_step(
                self.flow_results[t-1,:],
                self.flow_results[t,:],
                self.head_results[t-1,:],
                self.head_results[t,:],
                self.mesh.nodes_float[NODE_FLOAT['B'],:],
                self.mesh.nodes_float[NODE_FLOAT['R'],:])
            run_junction_step(
                self.mesh.junctions_int,
                self.flow_results[t-1,:],
                self.flow_results[t,:],
                self.head_results[t-1,:],
                self.head_results[t,:],
                self.mesh.nodes_float[NODE_FLOAT['B'],:],
                self.mesh.nodes_float[NODE_FLOAT['R'],:])
            run_reservoir_step(
                self.mesh.reservoir_ids,
                self.mesh.nodes_int[NODE_INT['is_start'],:],
                self.flow_results[t-1,:],
                self.flow_results[t,:],
                self.head_results[t-1,:],
                self.head_results[t,:],
                self.mesh.nodes_float[NODE_FLOAT['B'],:],
                self.mesh.nodes_float[NODE_FLOAT['R'],:])
            # run_valve_step()
            # run_pump_step()
        clk.toc()

    def _define_node_sim_constants(self):
        # ! ESTIMATE INITIAL CONDITIONS FIRST!!!
        for i in range(len(self.mesh.node_name_list)):
            link_id = self.mesh.nodes_int[NODE_INT['link_id'], i]

            wave_speed = self.mesh.links_float[LINK_FLOAT['wave_speed'], link_id]
            area = self.mesh.links_float[LINK_FLOAT['area'], link_id]

            ffactor = self.mesh.links_float[LINK_FLOAT['ffactor'], link_id]
            dx = self.mesh.links_float[LINK_FLOAT['dx'], link_id]
            diameter = self.mesh.links_float[LINK_FLOAT['diameter'], link_id]

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

        for pipe_name in self.mesh.wn.pipe_name_list:
            self.mesh.links_float[LINK_FLOAT['ffactor'], self.mesh.link_ids[pipe_name]] = float(self.steady_state_sim.link['frictionfact'][pipe_name])

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
        self.steady_state_sim = wntr.sim.EpanetSimulator(self.mesh.wn).run_sim()
        self.network_graph = self.wn.get_graph()
        self.mesh_graph = None
        self.time_step = None
        self.num_processors = None # number of processors

        # Number of segments are only defined for pipes
        self.segments = None

        # Wavespeed values associated to each link
        self.wave_speeds = None

        # Data structures associated to nodes (only boundary nodes and interior nodes)
        self.nodes_int = None
        self.nodes_float = None
        self.node_ids = None
        self.node_name_list = None
        # Boundary nodes
        self.reservoir_ids = None
        self.junction_ids = None
        self.valve_start_ids = None
        self.valve_end_ids = None
        self.pump_start_ids = None
        self.pump_end_ids = None

        # Data structures associated to junctions
        self.junctions_int = None
        self.junctions_float = None
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

        # * network_graph is a directed graph
        #  * There are sp-1 interior points for every pipe (sp is the
            number of segments in which a pipe is going to be divided
        # * Every pipe has 2 junction boundary nodes
        # * This function should be called only after defining the segments
            for each pipe in the network
        """

        visited_nodes = {}

        i = 0 # nodes index
        j = 0 # junctions index
        k = 0 # links index

        for start_node_name in self.network_graph:
            downstream_nodes = self.network_graph[start_node_name]

            downstream_link_names = [
                link for end_node_name in downstream_nodes
                for link in self.network_graph[start_node_name][end_node_name]
            ]

            # Junction definition
            # Update start_node as junction
            if start_node_name not in self.junction_ids:
                self.junction_ids[start_node_name] = j
                j += 1

            start_id = self.junction_ids[start_node_name] # start junction id
            self.junctions_int[JUNCTION_INT['downstream_neighbors_num'], start_id] = len(downstream_link_names)

            # Update downstream nodes
            for ii, downstream_node_name in enumerate(downstream_nodes):
                link_name = downstream_link_names[ii]
                link = self.wn.get_link(link_name)

                # Define downstream node as junction
                if downstream_node_name not in self.junction_ids:
                    self.junction_ids[downstream_node_name] = j
                    j += 1

                end_id = self.junction_ids[downstream_node_name] # end junction id

                if self.junctions_int[JUNCTION_INT['upstream_neighbors_num'], end_id] == NULL:
                    self.junctions_int[JUNCTION_INT['upstream_neighbors_num'], end_id] = 1
                else:
                    self.junctions_int[JUNCTION_INT['upstream_neighbors_num'], end_id] += 1

                # Index of last upstream node of end junction
                jj = len(self.network_graph[downstream_node_name]) + self.junctions_int[JUNCTION_INT['upstream_neighbors_num'], end_id] - 1

                # Link definition
                self.links_int[LINK_INT['id'], k] = k
                self.links_int[LINK_INT['link_type'], k] = LINK_TYPES[link.link_type]

                # Node definition
                if link.link_type == 'Pipe':
                    # * Nodes are stored in order, such that the i-1 and the i+1
                    #   correspond to the upstream and downstream nodes of the
                    #   i-th node
                    for idx in range(self.segments[link_name]+1):
                        # Pipe nodes definition
                        self.nodes_int[NODE_INT['id'], i] = i
                        self.nodes_int[NODE_INT['link_id'], i] = k
                        if idx == 0:
                            self.nodes_int[NODE_INT['node_type'], i] = NODE_TYPES['junction']
                            self.junctions_int[JUNCTION_INT['n%d' % ii], start_id] = i
                        elif idx == self.segments[link_name]:
                            self.nodes_int[NODE_INT['node_type'], i] = NODE_TYPES['junction']
                            self.junctions_int[JUNCTION_INT['n%d' % jj], end_id] = i
                        else:
                            self.nodes_int[NODE_INT['node_type'], i] = NODE_TYPES['interior']
                        pipe_diameter = link.diameter
                        pipe_area = (np.pi * link.diameter ** 2 / 4)
                        ffactor = float(self.steady_state_sim.link['frictionfact'][pipe_name])
                        dx = link.length / self.segments[link_name]
                        self.nodes_float[NODE_FLOAT['B'], i] = self.wave_speeds[link_name] / (G*pipe_area)
                        self.nodes_float[NODE_FLOAT['R'], i] = ffactor*dx / (2*G*pipe_diameter*pipe_area**2)
                        i += 1
                elif link.link_type in ('Valve', 'Pump'):
                    self.nodes_int[NODE_INT['id'], i] = i
                    self.nodes_int[NODE_INT['link_id'], i] = k
                    self.junctions_int[JUNCTION_INT['n%d' % ii], start_id] = i
                    self.junctions_int[JUNCTION_INT['n%d' % jj], end_id] = i
                    if link.link_type == 'Valve':
                        self.nodes_int[NODE_INT['node_type'], i] = NODE_TYPES['valve']
                    elif link.link_type == 'Pump':
                        self.nodes_int[NODE_INT['node_type'], i] = NODE_TYPES['pump']
                    i += 1
                else:
                    raise Exception("Link %s of type %s is not supported" % (start_links[0], link.link_type))
                k += 1

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

        self.junctions_int = np.full((len(JUNCTION_INT), len(self.network_graph)), NULL, dtype = int)
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

            self.junctions_int[JUNCTION_INT['upstream_neighbors_num'], i] = len(upstream)
            self.junctions_int[JUNCTION_INT['downstream_neighbors_num'], i] = len(downstream)

            j = 1
            for node_name in upstream:
                self.junctions_int[JUNCTION_INT['n%d' % j], i] = self.node_ids[node_name]
                self.junctions_int[JUNCTION_INT['p%d' % j], i] = self.get_processor(node_name)
                j += 1
            for node_name in downstream:
                self.junctions_int[JUNCTION_INT['n%d' % j], i] = self.node_ids[node_name]
                self.junctions_int[JUNCTION_INT['p%d' % j], i] = self.get_processor(node_name)
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