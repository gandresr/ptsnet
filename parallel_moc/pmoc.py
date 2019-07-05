import wntr
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import subprocess
import dis
import pickle

from constants import *
from scipy.interpolate import splev, splrep
from numba import njit, jit
from time import time
from os.path import isdir

np.set_printoptions(precision=2)

def save_model(mesh):
    """Saves mesh object

    Arguments:
        mesh {Mesh} -- mesh object
    """
    with open(mesh.fname + '.mesh', 'wb') as f:
        pickle.dump(vars(mesh), f)

def open_model(fname):
    """Returns pickled mesh

    Arguments:
        fname {string} -- mesh filemane

    Returns:
        [Mesh] -- stored mesh object
    """
    with open(fname[:fname.find('.')] + '.mesh', 'rb') as f:
        m = pickle.load(f)
    return Mesh(None, None, attrs=m)

# Parallel does not perform well in sandy-bridge architectures
@njit(parallel = True)
def run_interior_step(Q1, Q2, H1, H2, B, R):
    # Keep in mind that the first and last nodes in mesh.nodes will
    #   always be a boundary node
    for i in range(1, len(H1)-1):
        H2[i] = ((H1[i-1] + B[i]*Q1[i-1])*(B[i] + R[i]*abs(Q1[i+1])) \
            + (H1[i+1] - B[i]*Q1[i+1])*(B[i] + R[i]*abs(Q1[i-1]))) \
            / ((B[i] + R[i]*abs(Q1[i-1])) + (B[i] + R[i]*abs(Q1[i+1])))
        Q2[i] = ((H1[i-1] + B[i]*Q1[i-1]) - (H1[i+1] - B[i]*Q1[i+1])) \
            / ((B[i] + R[i]*abs(Q1[i-1])) + (B[i] + R[i]*abs(Q1[i+1])))

@njit(parallel = False)
def run_junction_step(
    t, junctions, B, R, Cm, Bm, Cp, Bp, Q1, H1, Q2, H2, demand, junction_type,
    RESERVOIR, PIPE, N, DOWN, UP):

    for j_id in range(junctions.shape[1]):

        downstream_num = junctions[DOWN, j_id]
        upstream_num = junctions[UP, j_id]

        # Junction is a reservoir
        if junction_type[j_id] == RESERVOIR:
            for j in range(downstream_num):
                k = junctions[j+N, j_id]
                H2[k] = H1[k]
                Q2[k] = (H1[k] - H1[k+1] + B[k+1]*Q1[k+1]) \
                        / (B[k+1] + R[k+1]*abs(Q1[k+1]))
            for j in range(downstream_num, upstream_num+downstream_num):
                k = junctions[j+N, j_id]
                H2[k] = H1[k]
                Q2[k] = (H1[k-1] + B[k-1]*Q1[k-1] - H1[k]) \
                        / (B[k-1] + R[k-1]*abs(Q1[k-1]))
        elif junction_type[j_id] == PIPE:
            sc = 0
            sb = 0

            for j in range(downstream_num): # C-
                k = junctions[j+N, j_id]
                Cm[k] = H1[k+1] - B[k+1]*Q1[k+1]
                Bm[k] = B[k+1] + R[k+1]*abs(Q1[k+1])
                sc += Cm[k] / Bm[k]
                sb += 1 / Bm[k]

            for j in range(downstream_num, upstream_num+downstream_num): # C+
                k = junctions[j+N, j_id]
                Cp[k] = H1[k-1] + B[k-1]*Q1[k-1]
                Bp[k] = B[k-1] + R[k-1]*abs(Q1[k-1])
                sc += Cp[k] / Bp[k]
                sb += 1 / Bp[k]

            HH = sc/sb + demand[j_id]/sb

            for j in range(downstream_num): # C-
                k = junctions[j+N, j_id]
                H2[k] = HH
                Q2[k] = (HH - Cm[k]) / Bm[k]

            for j in range(downstream_num, upstream_num+downstream_num): # C+
                k = junctions[j+N, j_id]
                H2[k] = HH
                Q2[k] = (Cp[k] - HH) / Bp[k]

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
        self.settings = []
        self.curves = []
        self.define_initial_conditions()

    def _check_model(self):
        for v in range(self.mesh.num_valves):
            start_id = self.mesh.valves_int[VALVE_INT['upstream_junction'], v]
            end_id = self.mesh.valves_int[VALVE_INT['downstream_junction'], v]
            unode = self.mesh.junctions_int[JUNCTION_INT['n2'], start_id]
            dnode = self.mesh.junctions_int[JUNCTION_INT['n1'], end_id]
            setting_id = self.mesh.valves_int[VALVE_INT['setting_id'], v]
            valve_name = self.mesh.valve_name_list[v]
            if setting_id == NULL:
                setting = self.mesh.valves_float[VALVE_FLOAT['setting'], v]
                if setting == NULL:
                    raise Exception('It is necessary to define a setting value for valve %s' % valve_name)
            if dnode != NULL and unode != NULL:
                curve_id = self.mesh.valves_int[VALVE_INT['curve_id'], v]
                if curve_id == NULL:
                    raise Exception("It is necessary to define a curve for valve %s" % valve_name)
        for p in range(self.mesh.num_pumps):
            setting_id = self.mesh.pumps_int[PUMP_INT['setting_id'], p]
            if setting_id == NULL:
                setting = self.mesh.pumps_float[PUMP_FLOAT['setting'], p]
                if setting == NULL:
                    pump_name = self.mesh.pump_name_list[v]
                    raise Exception('It is necessary to define a setting value for pump %s' % pump_name)

    def run_simulation(self):
        # clk = Clock()
        self._check_model()
        for t in range(1, self.time_steps):
            run_interior_step(
                self.flow_results[t-1,:],
                self.flow_results[t,:],
                self.head_results[t-1,:],
                self.head_results[t,:],
                self.mesh.nodes_float[NODE_FLOAT['B'],:],
                self.mesh.nodes_float[NODE_FLOAT['R'],:])
            run_junction_step(
                t, self.mesh.junctions_int,
                self.mesh.nodes_float[NODE_FLOAT['B'], :],
                self.mesh.nodes_float[NODE_FLOAT['R'], :],
                self.mesh.nodes_float[NODE_FLOAT['Cm'], :],
                self.mesh.nodes_float[NODE_FLOAT['Bm'], :],
                self.mesh.nodes_float[NODE_FLOAT['Cp'], :],
                self.mesh.nodes_float[NODE_FLOAT['Bp'], :],
                self.flow_results[t-1, :],
                self.head_results[t-1, :],
                self.flow_results[t, :],
                self.head_results[t, :],
                self.mesh.junctions_float[JUNCTION_FLOAT['demand'], :],
                self.mesh.junctions_int[JUNCTION_INT['junction_type'], :],
                JUNCTION_TYPES['reservoir'],
                JUNCTION_TYPES['pipes'],
                3,
                JUNCTION_INT['downstream_neighbors_num'],
                JUNCTION_INT['upstream_neighbors_num'])
            self.run_valve_step(t)

    def run_valve_step(self, t):
        B = self.mesh.nodes_float[NODE_FLOAT['B'], :]
        R = self.mesh.nodes_float[NODE_FLOAT['R'], :]
        Cm = self.mesh.nodes_float[NODE_FLOAT['Cm'], :]
        Bm = self.mesh.nodes_float[NODE_FLOAT['Bm'], :]
        Cp = self.mesh.nodes_float[NODE_FLOAT['Cp'], :]
        Bp = self.mesh.nodes_float[NODE_FLOAT['Bp'], :]
        Q0 = self.flow_results[0, :]
        Q1 = self.flow_results[t-1, :]
        H1 = self.head_results[t-1, :]
        Q2 = self.flow_results[t, :]
        H2 = self.head_results[t, :]
        for v in range(self.mesh.num_valves):
            start_id = self.mesh.valves_int[VALVE_INT['upstream_junction'], v]
            end_id = self.mesh.valves_int[VALVE_INT['downstream_junction'], v]
            unode = self.mesh.junctions_int[JUNCTION_INT['n2'], start_id]
            dnode = self.mesh.junctions_int[JUNCTION_INT['n1'], end_id]
            setting = self.mesh.valves_float[VALVE_FLOAT['setting'], v]
            setting_id = self.mesh.valves_int[VALVE_INT['setting_id'], v]
            curve_id = self.mesh.valves_int[VALVE_INT['curve_id'], v]
            area = self.mesh.valves_float[VALVE_FLOAT['area'], v]
            if setting == NULL:
                setting = self.settings[setting_id][t]
            else:
                self.mesh.valves_float[VALVE_FLOAT['setting'], v] = NULL
            if dnode == NULL:
                # End-valve
                Q2[unode] = Q0[unode] * setting
                H2[unode] = H1[unode-1] + B[unode-1]*Q1[unode-1] - (B[unode-1] + R[unode-1]*abs(Q1[unode-1]))*Q2[unode]
            else:
                # Inline-valve
                Cd = self.curves[curve_id](setting)
                Cp[unode] = H1[unode-1] + B[unode-1]*Q1[unode-1]
                Bp[unode] = B[unode-1] + R[unode-1]*abs(Q1[unode-1])
                Cm[dnode] = H1[dnode+1] - B[dnode+1]*Q1[dnode+1]
                Bm[dnode] = B[dnode+1] + R[dnode+1]*abs(Q1[dnode+1])
                S = -1 if (Cp[unode] - Cm[dnode]) < 0 else 1
                Cv = 2*G*(Cd*setting*area)**2
                X = Cv*(Bp[unode] + Bm[dnode])
                Q2[unode] = (-S*X + S*(X**2 + S*4*Cv*(Cp[unode] - Cm[dnode]))**0.5)/2
                Q2[dnode] = Q2[unode]
                H2[unode] = Cp[unode] - Bp[unode]*Q2[unode]
                H2[dnode] = Cm[dnode] + Bm[dnode]*Q2[dnode]

    def run_pump_step(self):
        pass

    def define_burst(self, junction_name, A, Cd):
        j_id = self.mesh.junction_ids[junction_name]
        pass

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

        link = self.mesh.wn.get_link(link_name)

        if curve is not None:
            cc = curve
        elif curve_file is not None:
            cc = np.loadtxt(curve_file, delimiter=',')

        curve_id = len(self.curves)

        if link.link_type in 'Valve':
            if curve_type != 'valve':
                raise Exception("Type of curve is not compatible with valve %s" % link_name)
            link_id = self.mesh.valve_ids[link_name]
            self.mesh.valves_int[VALVE_INT['curve_id'], link_id] = curve_id
        elif link.link_type in 'Pump':
            if curve_type != 'pump':
                raise Exception("Type of curve is not compatible with pump %s" % link_name)
            link_id = self.mesh.pump_ids[link_name]
            self.mesh.pumps_int[PUMP_INT['curve_id'], link_id] = curve_id

        cc = cc[np.argsort(cc[:,0])]
        X = cc[:,0]
        Y = cc[:,1]

        spl = splrep(X, Y)
        F = lambda x : splev(x, spl)

        self.curves.append(F)

    def define_valve_setting(self, valve_name, setting = None, setting_file = None):
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
        # TODO: UPDATE DOCS - default setting == ss[-1]
        """
        if setting is None and setting_file is None:
            raise Exception("It is necessary to define either a setting iterable or a setting_file")

        valve_id = self.mesh.valve_ids[valve_name]

        ss = []
        if setting is not None:
            if type(setting) == float:
                ss = setting
            else:
                ss = np.array(setting)
        elif setting_file is not None:
            ss = np.loadtxt(setting_file, dtype=float)

        setting_id = len(self.settings)

        N = len(ss)
        if N == 1:
            self.mesh.valves_float[VALVE_FLOAT['setting'], valve_id] = ss
        elif N < self.time_steps:
            ss = np.concatenate((ss, np.full((self.time_steps - N, 1), ss[-1])), axis = None)
        if N > 1:
            self.settings.append(ss[:self.time_steps])
            self.mesh.valves_int[VALVE_INT['setting_id'], valve_id] = setting_id

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

            if node_type not in  (NODE_TYPES['interior'], NODE_TYPES['boundary']):
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
    def __init__(self, input_file, dt, wave_speed_file = None, default_wave_speed = None, attrs = None):
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

        if attrs != None:
            self.__dict__.update(attrs)
            return

        self.fname = input_file[:input_file.find('.inp')]
        self.wn = wntr.network.WaterNetworkModel(input_file)
        self.steady_state_sim = wntr.sim.EpanetSimulator(self.wn).run_sim()
        self.network_graph = self._get_network_graph()

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
        self.num_valves = self.wn.num_valves

        # Data structures associated to junctions
        self.junctions_int = None
        self.junctions_float = None
        self.junction_ids = None
        self.junction_name_list = None
        self.num_junctions = 0

        # Data structures associated to links
        self.link_ids = None
        self.link_name_list = None

        # Data structures associated to valves
        self.valves_int = None
        self.valves_float = None
        self.valve_ids = None
        self.valve_name_list = None
        self.num_valves = 0

        # Data structures associated to pumps
        self.pumps_int = None
        self.pumps_float = None
        self.pump_ids = None
        self.pump_name_list = None
        self.num_pumps = 0

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

    def _get_network_graph(self):
        """[summary]

        Returns:
            [type] -- [description]
        """
        G = self.wn.get_graph()
        switch_links = []
        for n1 in G:
            for n2 in G[n1]:
                for link_name in G[n1][n2]:
                    flow = float(self.steady_state_sim.link['flowrate'][link_name])
                    if flow < -TOL:
                        switch_links.append((n1, n2))
                        self.steady_state_sim.link['flowrate'][link_name] *= -1
                    elif flow == 0:
                        ha = float(self.steady_state_sim.node['head'][n1])
                        hb = float(self.steady_state_sim.node['head'][n2])
                        if ha < hb:
                            switch_links.append((n1, n2))
        for n1, n2 in switch_links:
            attrs = G[n1][n2]
            link = list(attrs.keys())[0]
            G.add_edge(n2, n1, key=link, attr_dict=attrs[link])
            G.remove_edge(n1, n2)

        return G

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

    def _check_compatibility(self):
        """[summary]

        Raises:
            Exception: [description]
            Exception: [description]
            Exception: [description]
            Exception: [description]
            Exception: [description]
            Exception: [description]

        Assumptions:
        - Only 2 types of non-pipe elements: valves and pumps
        - The junctions of a non-pipe element can only be tanks, reservoirs or
            junctions (this includes dead-ends)
        - At most one link can be attached to a non-pipe element
        """

        for node in self.network_graph:
            d = self.network_graph.degree(node)
            dnodes = list(self.network_graph.successors(node))
            unodes = list(self.network_graph.predecessors(node))
            in_degree = len(unodes)
            out_degree = len(dnodes)
            if d == 0:
                raise Exception("Junction %s is isolated" % node)
            elif d == 1:
                if in_degree == 1:
                    link_name = list(self.network_graph[unodes[0]][node])[0]
                    link = self.wn.get_link(link_name)
                    if link.link_type == 'Pump':
                        raise Exception("Pump %s is not valid, it has a dead-end" % link_name)
                elif out_degree == 1:
                    if node not in self.wn.reservoir_name_list:
                        raise Exception("Junction %s can only be a reservoir" % node)
            elif d == 2:
                if in_degree == 1 and out_degree == 1:
                    u_link_name = list(self.network_graph[unodes[0]][node])[0]
                    d_link_name = list(self.network_graph[node][dnodes[0]])[0]
                    ulink = self.wn.get_link(u_link_name)
                    dlink = self.wn.get_link(d_link_name)
                    if ulink.link_type in ('Valve', 'Pump') and dlink.link_type in ('Valve', 'Pump'):
                        raise Exception("Links %s and %s are non-pipe elements and cannot be connected together" % (u_link_name, d_link_name))
            else:
                for i in range(in_degree):
                    n1 = unodes[i]
                    link_name = list(self.network_graph[n1][node])[0]
                    link = self.wn.get_link(link_name)
                    if link.link_type in ('Valve', 'Pump'):
                        raise Exception("Connection in junction is not valid, it can not have non-pipe elements ")
                for i in range(out_degree):
                    n2 = dnodes[i]
                    link_name = list(self.network_graph[node][n2])[0]
                    link = self.wn.get_link(link_name)
                    if link.link_type in ('Valve', 'Pump'):
                        raise Exception("Connection in junction is not valid, it can not have non-pipe elements ")

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

        num_total_nodes = sum(self.segments.values()) + len(self.segments)
        self.nodes_int = np.full(
            (len(NODE_INT), num_total_nodes), NULL, dtype = 'int')
        self.nodes_float = np.full(
            (len(NODE_FLOAT), num_total_nodes), NULL, dtype = 'float64')

        self.junctions_int = np.full(
            (len(JUNCTION_INT), self.wn.num_nodes), NULL, dtype = 'int')
        self.junctions_int[0:2,:] = 0
        self.junctions_float = np.full(
            (len(JUNCTION_FLOAT), self.wn.num_nodes), 0, dtype = 'float64')
        self.junction_ids = {}
        self.junction_name_list = []

        self.link_ids = {}
        self.link_name_list = []

        self.valves_int = np.full(
            (len(VALVE_INT), self.wn.num_valves) , NULL, dtype='int')
        self.valves_float = np.full(
            (len(VALVE_FLOAT), self.wn.num_valves), NULL, dtype='float64')
        self.valve_ids = {}
        self.valve_name_list = []

        self.pumps_int = np.full(
            (len(PUMP_INT), self.wn.num_pumps), NULL, dtype='int')
        self.pumps_float = np.full(
            (len(PUMP_FLOAT), self.wn.num_pumps), NULL, dtype='float64')
        self.pump_ids = {}
        self.pump_name_list = []

        i = 0 # nodes index
        j = 0 # junctions index
        k = 0 # links index
        p = 0 # pump index
        v = 0 # valve index

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
            # Define start junction demand
            self.junctions_float[JUNCTION_FLOAT['demand'], start_id] = float(self.steady_state_sim.node['demand'][start_node_name])
            # Check if start junction is a reservoir
            if start_node_name in self.wn.reservoir_name_list:
                self.junctions_float[JUNCTION_FLOAT['head'], start_id] = float(self.steady_state_sim.node['head'][start_node_name])
                self.junctions_int[JUNCTION_INT['junction_type'], start_id] = JUNCTION_TYPES['reservoir']
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

                # Set default junction types for start and end junctions
                if self.junctions_int[JUNCTION_INT['junction_type'], start_id] == NULL:
                    self.junctions_int[JUNCTION_INT['junction_type'], start_id] = JUNCTION_TYPES['pipes']
                if self.junctions_int[JUNCTION_INT['junction_type'], end_id] == NULL:
                    self.junctions_int[JUNCTION_INT['junction_type'], end_id] = JUNCTION_TYPES['pipes']

                # Define downstream junction demand
                self.junctions_float[JUNCTION_FLOAT['demand'], end_id] = float(self.steady_state_sim.node['demand'][downstream_node_name])
                # Check if downstream junction is a reservoir
                if downstream_node_name in self.wn.reservoir_name_list:
                    self.junctions_float[JUNCTION_FLOAT['head'], end_id] = float(self.steady_state_sim.node['head'][downstream_node_name])

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
                        self.nodes_int[NODE_INT['subindex'], i] = idx
                        self.nodes_int[NODE_INT['link_id'], i] = k
                        if idx == 0:
                            self.nodes_int[NODE_INT['node_type'], i] = NODE_TYPES['boundary']
                            self.junctions_int[JUNCTION_INT['n%d' % (ii+1)], start_id] = i
                        elif idx == self.segments[link_name]:
                            self.nodes_int[NODE_INT['node_type'], i] = NODE_TYPES['boundary']
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
                elif link.link_type == 'Valve':
                    self.junctions_int[JUNCTION_INT['junction_type'], start_id] = JUNCTION_TYPES['valve']
                    self.junctions_int[JUNCTION_INT['junction_type'], end_id] = JUNCTION_TYPES['valve']
                    self.valves_int[VALVE_INT['upstream_junction'], v] = start_id
                    self.valves_int[VALVE_INT['downstream_junction'], v] = end_id
                    self.valves_float[VALVE_FLOAT['area'], v] = (np.pi * link.diameter ** 2 / 4)
                    self.valve_ids[link_name] = v
                    self.valve_name_list.append(link_name)
                    v += 1
                elif link.link_type == 'Pump':
                    self.junctions_int[JUNCTION_INT['junction_type'], start_id] = JUNCTION_TYPES['pump']
                    self.junctions_int[JUNCTION_INT['junction_type'], end_id] = JUNCTION_TYPES['pump']
                    self.pumps_int[PUMP_INT['upstream_junction'], p] = start_id
                    self.pumps_int[PUMP_INT['downstream_junction'], p] = end_id
                    self.pump_ids[link_name] = p
                    self.pump_name_list.append(link_name)
                    p += 1
                # Link definition
                self.link_name_list.append(link_name)
                self.link_ids[link_name] = k
                k += 1

    def _write_mesh(self):
        pass

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
        self._check_compatibility()
        self._define_mesh()
        self.num_nodes = self.nodes_int.shape[1]
        self.num_junctions = self.junctions_int.shape[1]
        self.num_valves = self.valves_int.shape[1]
        self.num_pumps = self.pumps_int.shape[1]

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

clk = Clock()
