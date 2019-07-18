import wntr
import numpy as np

from phammer.simulation.constants import TOL, WARNINGS, G
from phammer.simulation.constants import POINTS_INT, POINTS_FLOAT
from phammer.simulation.constants import NODES_INT, NODES_FLOAT, NODES_OBJ
from phammer.simulation.constants import VALVES_INT, VALVES_FLOAT
from phammer.simulation.constants import PUMPS_INT, PUMPS_FLOAT
from phammer.simulation.constants import NODE_TYPES, POINT_TYPES

class Mesh:
    def __init__(self, input_file, time_step, wn, default_wave_speed = None, wave_speed_file = None, delimiter=','):

        self.steady_state_sim = None
        self.network_graph = None
        self.properties = {}

        self.num_segments = 0
        self.num_boundaries = 0
        self.num_points = 0
        self.num_nodes = 0
        self.num_valves = 0
        self.num_pumps = 0

        self.boundary_ids = []
        self.node_ids = {}
        self.valve_ids = {}
        self.pump_ids = {}

        self.wn = wn
        self.time_step = time_step

        self.wave_speeds = self._get_wave_speeds(default_wave_speed, wave_speed_file, delimiter)
        self.segments = self._get_segments(time_step)

        self._initialize()

    def _get_network_graph(self):
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
        max_dt = segments[min(segments, key=segments.get)]

        t_step = min(time_step, max_dt)
        if t_step != time_step and WARNINGS:
            print("Warning: Time step has been redefined to: %f [s]" % t_step)
        self.time_step = t_step

        # The number of segments is defined
        self.num_segments = 0
        for pipe in segments:
            segments[pipe] /= t_step
            int_segments = int(segments[pipe])
            # The wave_speed is adjusted to compensate the truncation error
            e = int_segments-segments[pipe] # truncation error
            self.wave_speeds[pipe] = self.wave_speeds[pipe]/(1 + e/segments[pipe])
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

        self.node_ids = {node : i for i, node in enumerate(self.wn.node_name_list)}
        self.valve_ids = {valve : i for i, valve in enumerate(self.wn.valve_name_list)}
        self.pump_ids = {pump : i for i, pump in enumerate(self.wn.pump_name_list)}

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

        self.properties['obj']['nodes'] =  NODES_OBJ(**{
            prop: [[] for i in range(self.num_nodes)] for prop in NODES_OBJ._fields
        })

    def create_mesh(self):
        self.steady_state_sim = wntr.sim.WNTRSimulator(self.wn).run_sim()
        self.network_graph = self._get_network_graph()

        i = 0 # nodes index

        for start_node in self.network_graph:

            start_node_id = self.node_ids[start_node]

            # Set default values for node_type
            self.properties['int']['nodes'].node_type[start_node_id] = NODE_TYPES['junction']

            downstream_nodes = self.network_graph[start_node]

            downstream_link_names = [
                link for end_node_name in downstream_nodes
                for link in self.network_graph[start_node][end_node_name]
            ]

            # Check if start junction is a reservoir
            if start_node in self.wn.reservoir_name_list:
                self.properties['int']['nodes'].node_type[start_node_id] = NODE_TYPES['reservoir']

            # Define start junction demand
            H0_start = float(self.steady_state_sim.node['head'][start_node])
            fixed_demand = float(self.steady_state_sim.node['demand'][start_node])
            emitter_demand = float(self.steady_state_sim.node['leak_demand'][start_node])
            self.properties['float']['nodes'].demand_coeff[start_node_id] = fixed_demand / (2*G*H0_start**0.5)
            self.properties['float']['nodes'].emitter_coeff[start_node_id] = emitter_demand / (2*G*H0_start**0.5)

            # Update downstream nodes
            for j, end_node in enumerate(downstream_nodes):
                end_node_id = self.node_ids[end_node]
                link_name = downstream_link_names[j]
                link = self.wn.get_link(link_name)
                # link_id is based on the wntr data structure
                link_id = self.steady_state_sim.link['status'].columns.get_loc(link_name)

                # Check if end junction is a reservoir
                if end_node in self.wn.reservoir_name_list:
                    self.properties['int']['nodes'].node_type[end_node_id] = NODE_TYPES['reservoir']

                # Define end junction demand
                H0_end = float(self.steady_state_sim.node['head'][end_node])
                fixed_demand = float(self.steady_state_sim.node['demand'][end_node])
                emitter_demand = float(self.steady_state_sim.node['leak_demand'][end_node])
                self.properties['float']['nodes'].demand_coeff[end_node_id] = fixed_demand / (2*G*H0_end)**0.5
                self.properties['float']['nodes'].emitter_coeff[end_node_id] = emitter_demand / (2*G*H0_end)**0.5

                if link.link_type == 'Pipe':
                    # Friction factor based on D-W equation
                    Q0 = float(self.steady_state_sim.link['flowrate'][link_name])
                    pipe_diameter = link.diameter
                    pipe_area = (np.pi * link.diameter ** 2 / 4)
                    pipe_length = link.length
                    ffactor = (abs(H0_start - H0_end)*2*G*pipe_diameter) / (pipe_length * (Q0 / pipe_area)**2)

                    #  Points are stored in order, such that the i-1 and the i+1 points
                    #  correspond to the upstream and downstream points of the
                    #  i-th point
                    for idx in range(self.segments[link_name]+1):
                        self.properties['int']['points'].subindex[i] = idx
                        self.properties['int']['points'].link_id[i] = link_id
                        if idx == 0: # downstream node of start_node
                            self.properties['int']['points'].point_type[i] = POINT_TYPES['boundary']
                            self.properties['obj']['nodes'].downstream_points[start_node_id].append(i)
                            self.boundary_ids.append(i)
                        elif idx == self.segments[link_name]: # upstream node of end_node
                            self.properties['int']['points'].point_type[i] = POINT_TYPES['boundary']
                            self.properties['obj']['nodes'].upstream_points[end_node_id].append(i)
                            self.boundary_ids.append(i)
                        else: # interior point
                            self.properties['int']['points'].point_type[i] = POINT_TYPES['interior']
                        dx = pipe_length / self.segments[link_name]
                        self.properties['float']['points'].B[i] = self.wave_speeds[link_name] / (G*pipe_area)
                        self.properties['float']['points'].R[i] = ffactor*dx / (2*G*pipe_diameter*pipe_area**2)
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
                    self.properties['int']['pumps'].upstream_junction[self.pump_ids[link_name]] = self.node_ids[start_node]
                    self.properties['int']['pumps'].downstream_junction[self.pump_ids[link_name]] = self.node_ids[end_node]
                    (a, b, c,) = link.get_head_curve_coefficients()
                    self.properties['float']['pumps'].a[self.pump_ids[link_name]] = a
                    self.properties['float']['pumps'].b[self.pump_ids[link_name]] = b
                    self.properties['float']['pumps'].c[self.pump_ids[link_name]] = c