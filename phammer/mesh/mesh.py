import numpy as np
import phammer.epanet.sim as ss

from time import time
from phammer.simulation.constants import TOL, TIMEIT, WARNINGS, G, DEFAULT_FFACTOR
from phammer.simulation.constants import POINTS_INT, POINTS_FLOAT
from phammer.simulation.constants import NODES_INT, NODES_FLOAT
from phammer.simulation.constants import VALVES_INT, VALVES_FLOAT
from phammer.simulation.constants import PUMPS_INT, PUMPS_FLOAT
from phammer.simulation.constants import POINT_SUBTYPES, POINT_TYPES

class Mesh:
    # Note: all numpy arrays are initialized as None
    def __init__(self, time_step, ss, default_wave_speed = None, wave_speed_file = None, delimiter=','):
        """[summary]

        Arguments:
            time_step {[type]} -- [description]
            ss {[type]} -- steady state solution

        Keyword Arguments:
            default_wave_speed {[type]} -- [description] (default: {None})
            wave_speed_file {[type]} -- [description] (default: {None})
            delimiter {str} -- [description] (default: {','})
        """

        # Steady state results
        self.ss = ss

        self.properties = {}

        self.num_segments = 0
        self.num_boundaries = 0
        self.num_points = 0

        # Table with points and IDs
        self.node_points = None

        self.time_step = time_step
        self.period = period
        self.period_size = 0

        self.wave_speeds = self._get_wave_speeds(default_wave_speed, wave_speed_file, delimiter)
        self.segments = self._get_segments(time_step)

        # Initial conditions
        self.Q0 = None
        self.H0 = None
        self._initialize()

    def _get_wave_speeds(self, default_wave_speed = None, wave_speed_file = None, delimiter=','):
        wave_speeds = {}

        if default_wave_speed is None and wave_speed_file is None:
            raise Exception("Wave speed values not specified")

        if default_wave_speed is not None:
            wave_speeds = dict.fromkeys(self.ss.pipes, default_wave_speed)

        if wave_speed_file:
            with open(wave_speed_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    pipe, wave_speed = line.split(delimiter)
                    wave_speeds[pipe] = float(wave_speed)

        if len(wave_speeds) != self.ss.num_pipes):
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
        segments = self.ss.get_pipes('length')

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

        self.num_boundaries = 2*len(self.ss.num_pipes)
        self.num_points = self.num_segments + len(self.segments)
        self.num_nodes = self.ss.num_nodes
        self.num_valves = self.ss.num_valves
        self.num_pumps = self.ss.num_pumps

        self.pboundary_ids = np.zeros(self.ss.num_pipes, dtype = np.int)
        self.mboundary_ids = np.zeros(self.ss.num_pipes, dtype = np.int)
        num_reservoir_points = 0
        for r in self.ss.reservoirs:
            num_reservoir_points += self.ss.num_neighbors(r)
        self.reservoir_ids = np.zeros(num_reservoir_points, dtype = np.int)
        self.valve_node_ids = np.zeros(2*self.num_valves, dtype = np.int)
        self.pump_node_ids = np.zeros(2*self.num_pumps, dtype = np.int)

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

        max_degree = max(dict(self.ss.num_max_neighbors).values())
        self.node_points = np.full((self.num_nodes, max_degree), -1, dtype = np.int)

    def create_mesh(self):
        self.ss.set_initial_conditions(self)
        self._define_flow_directions()
        self.Q0 = np.zeros(self.num_points, dtype = np.float)
        self.H0 = np.zeros(self.num_points, dtype = np.float)

        # Set default value for node_type
        self.properties['int']['nodes'].node_type.fill(NODE_TYPES['junction'])
        self.properties['int']['nodes'].num_upoints.fill(0)
        self.properties['int']['nodes'].num_dpoints.fill(0)
        self.properties['float']['points'].has_Cm.fill(1)
        self.properties['float']['points'].has_Cp.fill(1)

        i = 0 # points index

        for link in self.ss.links:
            link_id = self.ss.link_ids[link_name]
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
                            self.reservoir_ids[r_i] = i
                            r_i += 1
                        self.properties['int']['points'].point_type[i] = POINT_TYPES['boundary']
                        self.properties['float']['points'].has_Cp[i] = 0
                        num_dpoints = self.properties['int']['nodes'].num_dpoints[start_node_id]
                        num_upoints = self.properties['int']['nodes'].num_upoints[start_node_id]
                        self.node_points[start_node_id, num_dpoints+num_upoints] = i
                        self.properties['int']['nodes'].num_dpoints[start_node_id] += 1
                    elif idx == self.segments[link_name]: # upstream node of end_node
                        if self.properties['int']['nodes'].node_type[end_node_id] == NODE_TYPES['reservoir']:
                            self.reservoir_ids.append(i)
                        self.properties['int']['points'].point_type[i] = POINT_TYPES['boundary']
                        self.properties['float']['points'].has_Cm[i] = 0
                        num_dpoints = self.properties['int']['nodes'].num_dpoints[end_node_id]
                        num_upoints = self.properties['int']['nodes'].num_upoints[end_node_id]
                        self.node_points[end_node_id, num_dpoints+num_upoints] = i
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
                self.properties['int']['nodes'].node_type[start_node_id] = NODE_TYPES['valve']
                self.properties['int']['nodes'].node_type[end_node_id] = NODE_TYPES['valve']
                self.properties['int']['valves'].upstream_node[self.valve_ids[link_name]] = self.node_ids[start_node]
                self.properties['int']['valves'].downstream_node[self.valve_ids[link_name]] = self.node_ids[end_node]
                self.properties['float']['valves'].area[self.valve_ids[link_name]] = (np.pi * link.diameter ** 2 / 4)
            elif link.link_type == 'Pump':
                self.properties['int']['nodes'].node_type[start_node_id] = NODE_TYPES['pump']
                self.properties['int']['nodes'].node_type[end_node_id] = NODE_TYPES['pump']
                self.properties['int']['pumps'].upstream_node[self.pump_ids[link_name]] = self.node_ids[start_node]
                self.properties['int']['pumps'].downstream_node[self.pump_ids[link_name]] = self.node_ids[end_node]
                (a, b, c,) = link.get_head_curve_coefficients()
                self.properties['float']['pumps'].a[self.pump_ids[link_name]] = a
                self.properties['float']['pumps'].b[self.pump_ids[link_name]] = b
                self.properties['float']['pumps'].c[self.pump_ids[link_name]] = c

        self.jnode_ids = np.delete(np.arange(self.num_nodes), self.valve_node_ids)
        self.jnode_ids = np.delete(self.jnode_ids, self.pump_node_ids)
        self.jboundary_ids = self.node_points[self.jnode_ids, :]
        temp_node_points = np.copy(self.jboundary_ids)
        self.jboundary_ids = self.jboundary_ids[self.jboundary_ids != -1]
        temp_node_points[temp_node_points != -1] = 0
        reps = temp_node_points.shape[1] + temp_node_points.sum(axis = 1)
        self.head_reps = np.array([i for i in range(temp_node_points.shape[0]) for j in range(reps[i])], dtype=np.int)
        self.bindices = np.zeros(len(reps), dtype = np.int)
        self.bindices[1:] = np.cumsum(reps)[:-1]