import wntr
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import subprocess
import dis

from constants import *
from scipy.interpolate import splev, splrep
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

def run_valve_step():
    pass

# @njit
# def run_junction_step(junctions_int, junctions_float,
#     nodes_int, nodes_float, Q1, Q2, H1, H2, B, R, settings):
#     # junctions is a matrix that contains neighbors info for each boundary node
#     #   bare in mind that neighbors of boundary nodes are also boundary nodes!
#     for i in range(junctions_int.shape[1]):
#         if junctions_int[0, i] == 0 and junctions_int[1,i] == 0:
#             continue

#         sc = 0
#         sb = 0

#         # downstream_neighbors_num
#         for j in range(junctions_int[0,i]):
#             k = junctions_int[j+2,i]-1
#             if nodes_int[2, k] == 1: # node_type == junction
#                 sc += (H1[k] + B[k]*Q1[k]) / (B[k] + R[k]*abs(Q1[k]))
#                 sb += 1 / (B[k] + R[k]*abs(Q1[k]))
#             elif nodes_int[2, k] == 2: # valve node
#                 end_junction = nodes_int[1, k]
#                 if

#         # upstream_neighbors_num
#         for j in range(junctions_int[0,i], junctions[0,i]+junctions[1,i]):
#             k = junctions[j+2,i]+1
#             if nodes_int[2, k] == 1: # node_type == junction
#                 sc += (H1[k] - B[k]*Q1[k]) / (B[k] + R[k]*abs(Q1[k]))
#                 sb += 1 / (B[k] + R[k]*abs(Q1[k]))

#         for j in range(junctions[0,i]):
#             k = junctions[j+2,i]-1
#             H2[j] = sc/sb
#             Q2[j] = (H1[k] + B[k]*Q1[k] - H2[k]) / (B[k] + R[k]*abs(Q1[k]))

#         for j in range(junctions[0,i], junctions[0,i]+junctions[1,i]):
#             k = junctions[j+2,i]+1
#             H2[j] = sc/sb
#             Q2[j] = (H2[k] - H1[k] + B[k]*Q1[k]) / (B[k] + R[k]*abs(Q1[k]))

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
        self.time_steps = T
        self.mesh = mesh
        self.steady_state_sim = mesh.steady_state_sim
        self.flow_results = np.zeros((T, mesh.num_nodes), dtype='float64')
        self.head_results = np.zeros((T, mesh.num_nodes), dtype='float64')
        self.discharge_coefficients = []
        self.curves = []
        self.define_initial_conditions()

    def run_simulation(self):
        clk = Clock()

        for t in range(1, self.time_steps):
            if t == 2:
                clk.tic()
            # HECK YEAH! THIS THING RUNS AT WARP SPEED
            run_interior_step(
                self.flow_results[t-1,:],
                self.flow_results[t,:],
                self.head_results[t-1,:],
                self.head_results[t,:],
                self.mesh.nodes_float[NODE_FLOAT['B'],:],
                self.mesh.nodes_float[NODE_FLOAT['R'],:])
            # run_junction_step(
            #     self.mesh.junctions_int,
            #     self.flow_results[t-1,:],
            #     self.flow_results[t,:],
            #     self.head_results[t-1,:],
            #     self.head_results[t,:],
            #     self.mesh.nodes_float[NODE_FLOAT['B'],:],
            #     self.mesh.nodes_float[NODE_FLOAT['R'],:])
            # run_valve_step()
            # run_pump_step()
        clk.toc()

    def define_curve(self, link_name, curve_type, curve = None, curve_file = None):
        """Defines curve values for a link

        If curve_type == 'Valve', then it corresponds to a Discharge
        coefficient vs setting. If a curve is defined, it should be
        a 2D array such that the first column contains setting values
        and the second column discharge coefficient values (it should be
        similarly done if a CSV file is defined)

        e.g.,

        curve = [[0.8, 1],
                 [0.6, 0.55],
                 [0.4, 0.28],
                 [0.2, 0.1],
                 [0, 0]]

        => setting = [0.8, 0.6, 0.4, 0.2, 0]
        => discharge_coefficients = [1, 0.55, 0.28, 0.1, 0]

        Arguments:
            link_name {string} -- [description]
            curve_type {string} -- [description]

        Keyword Arguments:
            curve {2D array} -- 2D array with curve values (default: {None})
            curve_file {string} -- path to curve file (default: {None})

        Raises:
            Exception: If curve or curve_file is not defined
            Exception: If curve_type is not compatible with link type
        """

        if curve is None and curve_file is None:
            raise Exception("It is necessary to define either a curve iterable or a curve_file")

        link_id = self.mesh.link_ids[link_name]

        if self.mesh.links_int[LINK_INT['link_type'], link_id] == LINK_TYPES['Valve']:
            if curve_type != 'Valve':
                raise Exception("Type of curve is not compatible with valve %s" % link_name)

        # TODO: Incorporate other types of curves

        if curve is not None:
            cc = np.array(curve, dtype='float64')
        elif curve_file is not None:
            cc = np.loadtxt(curve_file, delimiter=',')

        cc = cc[np.argsort(cc[:,0])] # order by x-axis
        self.curves.append(cc)

        curve_id = len(self.curves) - 1
        self.mesh.links_int[LINK_INT['curve_id'], link_id] = curve_id
        self.mesh.links_int[LINK_INT['curve_type'], link_id] = CURVE_TYPES[curve_type]

    def define_valve_setting(self, valve_name, setting = None, setting_file = None, default_setting = 1):
        """Defines setting values for a valve during the simulation time

        If the valve setting is not defined for a certain time step, the
        default_setting value will be used. If both, a vector and a file
        are defined, then, the vector is choosen and the file omitted.

        Arguments:
            valve_name {string} -- valve name as defined in EPANET

        Keyword Arguments:
            setting {iterable} -- float iterable with setting values for the
                first T steps of the simulation, T <= self.time_steps
                (default: {None})
            setting_file {string} -- path to file with setting values  for the
                first T steps of the simulation, T <= self.time_steps. The i-th
                line of the file has the value of the valve setting at the i-th
                time step (default: {None})
            default_setting {int} -- default value assigned to not setting
                values that are not defined (default: {1})

        Raises:
            Exception: If setting iterable or setting file is not defined
            Exception: If valve curve has not been defined
        """
        if setting is None and setting_file is None:
            raise Exception("It is necessary to define either a setting iterable or a setting_file")

        valve_id = self.mesh.link_ids[valve_name]
        curve_id = self.mesh.links_int[LINK_INT['curve_id'], valve_id]

        if curve_id == NULL:
            raise Exception("It is necessary to define the valve curve first")

        ss = []
        if setting is not None:
            ss = np.array(setting)
        elif setting_file is not None:
            ss = np.loadtxt(setting_file, dtype=float)

        N = len(ss)
        if N < self.time_steps:
            ss = np.concatenate((ss, np.full((self.time_steps - N, 1), default_setting)), axis = None)

        spl = splrep(self.curves[curve_id][:,0], self.curves[curve_id][:,1])
        # [:self.time_steps] in case that len(ss) > self.time_steps
        dcoeffs = splev(ss[:self.time_steps], spl)

        self.discharge_coefficients.append(dcoeffs)

        dcoeff_id = len(self.discharge_coefficients) - 1
        self.mesh.links_int[LINK_INT['dcoeff_id'], valve_id] = dcoeff_id

    def define_initial_conditions(self):
        """Extracts initial conditions from EPANET
        """
        for i in range(self.mesh.num_nodes):
            link_id = self.mesh.nodes_int[NODE_INT['link_id'], i]
            link_name = self.mesh.link_name_list[link_id]
            link = self.mesh.wn.get_link(link_name)

            self.flow_results[0][i] = float(self.steady_state_sim.link['flowrate'][link_name])
            start_node_name = link.start_node_name
            end_node_name = link.end_node_name
            k = self.mesh.nodes_int[NODE_INT['subindex'], i]
            node_type = self.mesh.nodes_int[NODE_INT['node_type'], i]

            if node_type not in  (NODE_TYPES['interior'], NODE_TYPES['junction']):
                self.head_results[0][i] = float(self.steady_state_sim.node['head'][start_node_name])
            else:
                head_1 = float(self.steady_state_sim.node['head'][start_node_name])
                head_2 = float(self.steady_state_sim.node['head'][end_node_name])
                dx = k * link.length / self.mesh.segments[link_name]
                if head_1 > head_2:
                    hl = head_1 - head_2
                    self.head_results[0][i] = head_1 - hl*dx/link.length
                else:
                    hl = head_2 - head_1
                    self.head_results[0][i] = head_1 + hl*dx/link.length

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
        self.steady_state_sim = wntr.sim.EpanetSimulator(self.wn).run_sim()
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
        self.num_nodes = 0

        # Data structures associated to junctions
        self.junctions_int = None
        self.junctions_float = None
        self.junction_ids = None
        self.junction_name_list = None
        self.num_junctions = 0

        # Data structures associated to links (pipes, valves, pumps)
        self.links_int = None
        self.links_float = None
        self.link_ids = None
        self.link_name_list = None
        self.num_links = 0

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
        # * There are sp-1 interior points for every pipe (sp is the
            number of segments in which a pipe is going to be divided
        # * Every pipe has 2 boundary nodes
        # * Every non-pipe element has 1 boundary node
        # * This function should only be called after defining the segments
            for each pipe in the network
        """

        num_total_nodes = \
            sum(self.segments.values()) + len(self.segments) \
                + self.wn.num_pumps + self.wn.num_valves
        self.nodes_int = np.full(
            (len(NODE_INT), num_total_nodes), NULL, dtype = 'int')
        self.nodes_float = np.full(
            (len(NODE_FLOAT), num_total_nodes), NULL, dtype = 'float64')

        self.links_int = np.full(
            (len(LINK_INT), self.wn.num_links), NULL, dtype = 'int')
        self.links_float = np.full(
            (len(LINK_FLOAT), self.wn.num_links), NULL, dtype = 'float64')
        self.link_ids = {}
        self.link_name_list = []

        self.junctions_int = np.full(
            (len(JUNCTION_INT), self.wn.num_nodes), NULL, dtype = 'int')
        self.junctions_float = np.full(
            (len(JUNCTION_FLOAT), self.wn.num_nodes), NULL, dtype = 'float64')
        self.junction_ids = {}
        self.junction_name_list = []

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
                self.junction_name_list.append(start_node_name)
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
                    self.junction_name_list.append(downstream_node_name)
                    j += 1

                end_id = self.junction_ids[downstream_node_name] # end junction id

                if self.junctions_int[JUNCTION_INT['upstream_neighbors_num'], end_id] == NULL:
                    self.junctions_int[JUNCTION_INT['upstream_neighbors_num'], end_id] = 1
                else:
                    self.junctions_int[JUNCTION_INT['upstream_neighbors_num'], end_id] += 1

                # Index of last upstream node of end junction
                jj = len(self.network_graph[downstream_node_name]) + self.junctions_int[JUNCTION_INT['upstream_neighbors_num'], end_id] - 1

                # Node definition
                if link.link_type == 'Pipe':
                    # * Nodes are stored in order, such that the i-1 and the i+1
                    #   correspond to the upstream and downstream nodes of the
                    #   i-th node
                    for idx in range(self.segments[link_name]+1):
                        # Pipe nodes definition
                        self.nodes_int[NODE_INT['id'], i] = i
                        self.nodes_int[NODE_INT['subindex'], i] = idx
                        self.nodes_int[NODE_INT['link_id'], i] = k
                        if idx == 0:
                            self.nodes_int[NODE_INT['node_type'], i] = NODE_TYPES['junction']
                            self.junctions_int[JUNCTION_INT['n%d' % (ii+1)], start_id] = i
                        elif idx == self.segments[link_name]:
                            self.nodes_int[NODE_INT['node_type'], i] = NODE_TYPES['junction']
                            self.junctions_int[JUNCTION_INT['n%d' % (jj+1)], end_id] = i
                        else:
                            self.nodes_int[NODE_INT['node_type'], i] = NODE_TYPES['interior']
                        pipe_diameter = link.diameter
                        pipe_area = (np.pi * link.diameter ** 2 / 4)
                        ffactor = float(self.steady_state_sim.link['frictionfact'][link_name])
                        dx = link.length / self.segments[link_name]
                        self.nodes_float[NODE_FLOAT['B'], i] = self.wave_speeds[link_name] / (G*pipe_area)
                        self.nodes_float[NODE_FLOAT['R'], i] = ffactor*dx / (2*G*pipe_diameter*pipe_area**2)
                        i += 1
                elif link.link_type in ('Valve', 'Pump'):
                    self.nodes_int[NODE_INT['id'], i] = i
                    self.nodes_int[NODE_INT['link_id'], i] = k
                    self.nodes_int[NODE_INT['subindex'], i] = end_id
                    self.junctions_int[JUNCTION_INT['n%d' % (ii+1)], start_id] = i
                    self.junctions_int[JUNCTION_INT['upstream_neighbors_num'], end_id] -= 1
                    if link.link_type == 'Valve':
                        self.nodes_int[NODE_INT['node_type'], i] = NODE_TYPES['valve']
                    elif link.link_type == 'Pump':
                        self.nodes_int[NODE_INT['node_type'], i] = NODE_TYPES['pump']
                    i += 1
                else:
                    raise Exception("Link %s of type %s is not supported" % (link_name, link.link_type))

                # Link definition
                self.links_int[LINK_INT['id'], k] = k
                self.links_int[LINK_INT['link_type'], k] = LINK_TYPES[link.link_type]
                self.link_name_list.append(link_name)
                self.link_ids[link_name] = k
                k += 1

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
        self.num_nodes = self.nodes_int.shape[1]
        self.num_links = self.links_int.shape[1]
        self.num_junctions = self.junctions_int.shape[1]

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