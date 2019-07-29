import wntr
import numpy as np
import networkx as nx

from time import time
from phammer.simulation.constants import TOL, TIMEIT, WARNINGS, G, DEFAULT_FFACTOR
from phammer.simulation.constants import POINTS_INT, POINTS_FLOAT
from phammer.simulation.constants import NODES_INT, NODES_FLOAT
from phammer.simulation.constants import VALVES_INT, VALVES_FLOAT
from phammer.simulation.constants import PUMPS_INT, PUMPS_FLOAT
from phammer.simulation.constants import NODE_TYPES, POINT_TYPES

class Mesh:
    def __init__(self, input_file, time_step, wn, period = 0, default_wave_speed = None, wave_speed_file = None, delimiter=','):

        # Steady state results
        self.steady_head = None
        self.steady_leak_demand = None
        self.steady_demand = None
        self.steady_flowrate = None

        self.properties = {}

        self.num_segments = 0
        self.num_boundaries = 0
        self.num_points = 0
        self.num_nodes = 0
        self.num_valves = 0
        self.num_pumps = 0

        # IDs of boundary nodes at junctions
        self.pboundary_ids = [] # Boundaries with C+
        self.mboundary_ids = [] # Boundaries with C-
        self.jboundary_ids = [] # All boundary nodes at junctions
        self.valve_node_ids = []
        self.pump_node_ids = []
        # IDs of reservoirs
        self.reservoir_ids = []
        self.head_reps = [] # TODO explain
        self.bindices = [] # TODO explain
        self.node_points = None
        self.flow_directions = {}
        self.node_ids = {}
        self.valve_ids = {}
        self.pump_ids = {}

        self.wn = wn
        self.time_step = time_step
        self.period = period
        self.period_size = 0

        self.network_graph = self.wn.get_graph()
        self.wave_speeds = self._get_wave_speeds(default_wave_speed, wave_speed_file, delimiter)
        self.segments = self._get_segments(time_step)

        # Initial conditions
        self.Q0 = None
        self.H0 = None
        self._initialize()

    def _define_flow_directions(self):

        zero_graph = nx.Graph()

        k = 0
        j = 0

        for n1 in self.network_graph:
            for n2 in self.network_graph[n1]:
                for link_name in self.network_graph[n1][n2]:
                    flow = float(self.steady_flowrate[self.link_ids[link_name]])
                    if flow < -TOL:
                        self.steady_flowrate[link_name] *= -1
                        self.flow_directions[link_name] = -1
                        k += 1
                    elif flow > TOL:
                        self.flow_directions[link_name] = 1
                        k += 1
                    else:
                        self.steady_flowrate[link_name] = 0
                        link = self.wn.get_link(link_name)
                        zero_graph.add_edge(link.start_node_name, link.end_node_name)

        # Define flow convention for links with zero flow
        for n1, n2 in nx.dfs_edges(zero_graph):
            if not self.network_graph.has_edge(n1, n2):
                for link_name in self.network_graph[n2][n1]:
                    self.flow_directions[link_name] = -1
            else:
                for link_name in self.network_graph[n1][n2]:
                    j += 1
                    self.flow_directions[link_name] = 1

    def _get_wave_speeds(self, default_wave_speed = None, wave_speed_file = None, delimiter=','):
        wave_speeds = {}

        if default_wave_speed is None and wave_speed_file is None:
            raise Exception("Wave speed values not specified")

        if default_wave_speed is not None:
            wave_speeds = dict.fromkeys(self.wn.pipe_name_list, default_wave_speed)

        if wave_speed_file:
            with open(wave_speed_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    pipe, wave_speed = line.split(delimiter)
                    wave_speeds[pipe] = float(wave_speed)

        if len(wave_speeds) != len(self.wn.pipe_name_list):
            wave_speeds = {}
            raise Exception("""
            The file does not specify wave speed values for all the pipes,
            it is necessary to define a default wave speed value""")

        return wave_speeds

    def _get_segments(self, time_step):
        """Estimates the number of segments for each pipe in the EPANET network

        Pipes are segmented in order to create a Mesh for the MOC

        Arguments:
            dt {float} -- desired time step
        """
        # Get the maximum time steps for each pipe
        segments = self.wn.query_link_attribute('length')

        for pipe in segments:
            segments[pipe] /= self.wave_speeds[pipe]

        # Maximum time_step in the system to capture waves in all pipes
        max_dt = segments[min(segments, key=segments.get)] / 2 # at least 2 segments in critical pipe

        t_step = min(time_step, max_dt)
        if t_step != time_step and WARNINGS:
            print("Warning: Time step has been redefined to: %f [s]" % t_step)
        self.time_step = t_step

        # The number of segments is defined
        self.num_segments = 0
        for pipe in segments:
            segments[pipe] /= t_step
            int_segments = round(segments[pipe])
            # The wave_speed is adjusted to compensate the truncation error
            self.wave_speeds[pipe] = self.wave_speeds[pipe]*segments[pipe]/int_segments
            self.num_segments += int_segments
            segments[pipe] = int_segments

        return segments

    def _initialize(self):
        self.properties = {'int': {}, 'float': {}, 'obj': {}}

        self.num_boundaries = 2*self.wn.num_pipes
        self.num_points = self.num_segments + len(self.segments)
        self.num_nodes = self.wn.num_nodes
        self.num_valves = self.wn.num_valves
        self.num_pumps = self.wn.num_pumps

        self.link_ids = {link : i for i, link in enumerate(self.wn.link_name_list)}
        self.flow_directions = {}
        self.node_ids = {node : i for i, node in enumerate(self.wn.node_name_list)}
        self.valve_ids = {valve : i for i, valve in enumerate(self.wn.valve_name_list)}
        self.pump_ids = {pump : i for i, pump in enumerate(self.wn.pump_name_list)}
        self.junction_node_ids = set(range(self.num_nodes))

        self.properties['int']['points'] = POINTS_INT(**{
            prop: np.full(self.num_points, np.nan, dtype = np.int) for prop in POINTS_INT._fields
        })
        self.properties['int']['nodes'] = NODES_INT(**{
            prop: np.full(self.num_nodes, np.nan, dtype = np.int) for prop in NODES_INT._fields
        })
        self.properties['int']['valves'] = VALVES_INT(**{
            prop: np.full(self.num_valves, np.nan, dtype = np.int) for prop in VALVES_INT._fields
        })
        self.properties['int']['pumps'] = PUMPS_INT(**{
            prop: np.full(self.num_pumps, np.nan, dtype = np.int) for prop in PUMPS_INT._fields
        })

        self.properties['float']['points'] = POINTS_FLOAT(**{
            prop: np.zeros(self.num_points, dtype = np.float) for prop in POINTS_FLOAT._fields
        })
        self.properties['float']['nodes'] =  NODES_FLOAT(**{
            prop: np.zeros(self.num_nodes, dtype = np.float) for prop in NODES_FLOAT._fields
        })
        self.properties['float']['valves'] = VALVES_FLOAT(**{
            prop: np.zeros(self.num_valves, dtype = np.float) for prop in VALVES_FLOAT._fields
        })
        self.properties['float']['pumps'] =  PUMPS_FLOAT(**{
            prop: np.zeros(self.num_pumps, dtype = np.float) for prop in PUMPS_FLOAT._fields
        })

        max_degree = max(dict(self.network_graph.degree()).values())
        self.node_points = np.full((self.num_nodes, max_degree), -1, dtype = np.int)

    def _run_steady_state(self):
        if TIMEIT:
            t = time()
            print("START - EPANET")

        steady_state_results = wntr.sim.EpanetSimulator(self.wn).run_sim()

        if TIMEIT:
            print("END - EPANET [%.3f s]" % (time() - t))

        # Check if period is valid
        if steady_state_results.link['flowrate'].shape[0] < 2:
            if self.period >= 1:
                raise Exception("Not valid period")
        else:
            self.period_size = steady_state_results.link['flowrate'].index[1]

        # fix leak_demand
        self.steady_leak_demand = steady_state_results.node['demand'].loc[self.period_size*self.period]
        self.steady_head = steady_state_results.node['head'].loc[self.period_size*self.period]
        self.steady_demand = steady_state_results.node['demand'].loc[self.period_size*self.period]
        self.steady_flowrate = steady_state_results.link['flowrate'].loc[self.period_size*self.period]
        self._define_flow_directions()

    def create_mesh(self):
        self._run_steady_state()
        self.Q0 = np.zeros(self.num_points, dtype = np.float)
        self.H0 = np.zeros(self.num_points, dtype = np.float)

        # Set default value for node_type
        self.properties['int']['nodes'].node_type.fill(NODE_TYPES['junction'])
        self.properties['int']['nodes'].num_upoints.fill(0)
        self.properties['int']['nodes'].num_dpoints.fill(0)
        self.properties['float']['points'].is_mboundary.fill(1)
        self.properties['float']['points'].is_pboundary.fill(1)

        i = 0 # points index

        for link_name, link in self.wn.links():
            link_id = self.link_ids[link_name]
            start_node = link.start_node_name
            end_node = link.end_node_name
            if self.flow_directions[link_name] == -1:
                start_node, end_node = end_node, start_node
            start_node_id = self.node_ids[start_node]
            end_node_id = self.node_ids[end_node]

            # Check if start junction is a reservoir
            if start_node in self.wn.reservoir_name_list:
                self.properties['int']['nodes'].node_type[start_node_id] = NODE_TYPES['reservoir']
            if end_node in self.wn.reservoir_name_list:
                self.properties['int']['nodes'].node_type[end_node_id] = NODE_TYPES['reservoir']

            # Define start junction demand
            H0_start = float(self.steady_head[start_node])
            fixed_demand = float(self.steady_demand[start_node])
            emitter_demand = float(self.steady_leak_demand[start_node])
            self.properties['float']['nodes'].demand_coeff[start_node_id] = fixed_demand / (2*G*H0_start**0.5)
            self.properties['float']['nodes'].emitter_coeff[start_node_id] = emitter_demand / (2*G*H0_start**0.5)

            # Define end junction demand
            H0_end = float(self.steady_head[end_node])
            fixed_demand = float(self.steady_demand[end_node])
            emitter_demand = float(self.steady_leak_demand[end_node])
            self.properties['float']['nodes'].demand_coeff[end_node_id] = fixed_demand / (2*G*H0_end)**0.5
            self.properties['float']['nodes'].emitter_coeff[end_node_id] = emitter_demand / (2*G*H0_end)**0.5

            if link.link_type == 'Pipe':
                # Friction factor based on D-W equation
                Q0 = float(self.steady_flowrate[link_name])
                pipe_diameter = link.diameter
                pipe_area = (np.pi * link.diameter ** 2 / 4)
                pipe_length = link.length
                if Q0 < TOL:
                    ffactor = DEFAULT_FFACTOR
                else:
                    ffactor = (abs(H0_start - H0_end)*2*G*pipe_diameter) / (pipe_length * (Q0 / pipe_area)**2)

                #  Points are stored in order, such that the i-1 and the i+1 points
                #  correspond to the upstream and downstream points of the
                #  i-th point
                for idx in range(self.segments[link_name]+1):
                    self.properties['int']['points'].subindex[i] = idx
                    self.properties['int']['points'].link_id[i] = link_id
                    if idx == 0: # downstream node of start_node
                        if self.properties['int']['nodes'].node_type[start_node_id] == NODE_TYPES['reservoir']:
                            self.reservoir_ids.append(i)
                        self.properties['int']['points'].point_type[i] = POINT_TYPES['boundary']
                        num_dpoints = self.properties['int']['nodes'].num_dpoints[start_node_id]
                        num_upoints = self.properties['int']['nodes'].num_upoints[start_node_id]
                        self.node_points[start_node_id, num_dpoints+num_upoints] = i
                        self.properties['float']['points'].is_pboundary[i] = 0
                        self.mboundary_ids.append(i)
                        self.properties['int']['nodes'].num_dpoints[start_node_id] += 1
                    elif idx == self.segments[link_name]: # upstream node of end_node
                        if self.properties['int']['nodes'].node_type[end_node_id] == NODE_TYPES['reservoir']:
                            self.reservoir_ids.append(i)
                        self.properties['int']['points'].point_type[i] = POINT_TYPES['boundary']
                        num_dpoints = self.properties['int']['nodes'].num_dpoints[end_node_id]
                        num_upoints = self.properties['int']['nodes'].num_upoints[end_node_id]
                        self.node_points[end_node_id, num_dpoints+num_upoints] = i
                        self.properties['float']['points'].is_mboundary[i] = 0
                        self.pboundary_ids.append(i)
                        self.properties['int']['nodes'].num_upoints[end_node_id] += 1
                    else: # interior point
                        self.properties['int']['points'].point_type[i] = POINT_TYPES['interior']
                    dx = pipe_length / self.segments[link_name]
                    self.properties['float']['points'].B[i] = self.wave_speeds[link_name] / (G*pipe_area)
                    self.properties['float']['points'].R[i] = ffactor*dx / (2*G*pipe_diameter*pipe_area**2)
                    self.Q0[i] = float(self.steady_flowrate[link_name])
                    head_1 = float(self.steady_head[start_node])
                    head_2 = float(self.steady_head[end_node])
                    self.H0[i] = head_1 - (head_1 - head_2)*idx/self.segments[link_name]
                    i += 1
            elif link.link_type == 'Valve':
                self.valve_node_ids += [start_node_id, end_node_id]
                self.properties['int']['nodes'].node_type[start_node_id] = NODE_TYPES['valve']
                self.properties['int']['nodes'].node_type[end_node_id] = NODE_TYPES['valve']
                self.properties['int']['valves'].upstream_node[self.valve_ids[link_name]] = self.node_ids[start_node]
                self.properties['int']['valves'].downstream_node[self.valve_ids[link_name]] = self.node_ids[end_node]
                self.properties['float']['valves'].area[self.valve_ids[link_name]] = (np.pi * link.diameter ** 2 / 4)
            elif link.link_type == 'Pump':
                self.pump_node_ids += [start_node_id, end_node_id]
                self.properties['int']['nodes'].node_type[start_node_id] = NODE_TYPES['pump']
                self.properties['int']['nodes'].node_type[end_node_id] = NODE_TYPES['pump']
                self.properties['int']['pumps'].upstream_node[self.pump_ids[link_name]] = self.node_ids[start_node]
                self.properties['int']['pumps'].downstream_node[self.pump_ids[link_name]] = self.node_ids[end_node]
                (a, b, c,) = link.get_head_curve_coefficients()
                self.properties['float']['pumps'].a[self.pump_ids[link_name]] = a
                self.properties['float']['pumps'].b[self.pump_ids[link_name]] = b
                self.properties['float']['pumps'].c[self.pump_ids[link_name]] = c

        self.jnode_ids = np.delete(np.arange(self.num_nodes), self.valve_node_ids + self.pump_node_ids)
        self.jboundary_ids = self.node_points[self.jnode_ids, :]
        temp_node_points = np.copy(self.jboundary_ids)
        self.jboundary_ids = self.jboundary_ids[self.jboundary_ids != -1]
        temp_node_points[temp_node_points != -1] = 0
        reps = temp_node_points.shape[1] + temp_node_points.sum(axis = 1)
        print(reps)
        self.head_reps = [i for i in range(temp_node_points.shape[0]) for j in range(reps[i])]
        self.bindices = np.zeros(len(reps), dtype = np.int)
        self.bindices[1:] = np.cumsum(reps)[:-1]
        print(self.bindices)