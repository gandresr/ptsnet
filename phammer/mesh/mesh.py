import wntr
from numpy import isnan, nan

from .constants import *
from ..arrays.arrays import define_data_table

class Mesh:
    def __init__(self, input_file, time_step, default_wave_speed = None, wave_speed_file = None, delimiter=','):
        self.wn = wntr.network.WaterNetworkModel(input_file)
        self.steady_state_sim = wntr.sim.EpanetSimulator(self.wn).run_sim()
        self.network_graph = self.get_network_graph()
        self.time_step = time_step
        self.wave_speeds = self.get_wave_speeds(default_wave_speed = None, wave_speed_file = None, delimiter=',')
        self.segments = self.get_segments(time_step)
        self.properties = self.define_properties()

    def get_network_graph(self):
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

    def get_wave_speeds(self, default_wave_speed = None, wave_speed_file = None, delimiter=','):
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

    def get_segments(self, time_step):
        """Estimates the number of segments for each pipe in the EPANET network

        Pipes are segmented in order to create a Mesh for the MOC

        Arguments:
            dt {float} -- desired time step
        """
        # Get the maximum time steps for each pipe
        segments = self.wn.query_link_attribute('length') # The length attribute is just for pipes

        for pipe in segments:
            segments[pipe] /= self.wave_speeds[pipe]

        # Maximum time_step in the system to capture waves in all pipes
        max_dt = segments[min(segments, key=segments.get)]

        # Desired dt < max_dt ?
        t_step = min(time_step, max_dt)
        if t_step != time_step:
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

    def define_properties(self):
        properties = {'int': {}, 'float': {}, 'obj': {}}

        self.num_points = self.num_segments + len(self.segments)
        self.num_nodes = self.wn.num_nodes
        self.num_valves = self.wn.num_valves
        self.num_pumps = self.wn.num_pumps

        properties['int']['points'] = define_data_table(
            range(self.num_points), POINTS_INT)
        properties['int']['nodes'] = define_data_table(
            range(self.num_points), NODES_INT)
        properties['int']['valves'] = define_data_table(
            range(self.num_points), VALVES_INT)
        properties['int']['pumps'] = define_data_table(
            range(self.num_points), PUMPS_INT)

        properties['float']['points'] = define_data_table(
            range(self.num_points), POINTS_FLOAT)
        properties['float']['nodes'] = define_data_table(
            self.wn.node_name_list, NODES_FLOAT)
        properties['float']['valves'] = define_data_table(
            self.wn.valve_name_list, VALVES_FLOAT)
        properties['float']['pumps'] = define_data_table(
            self.wn.pump_name_list, PUMPS_FLOAT)

        properties['obj']['nodes'] = define_data_table(
            self.wn.node_name_list, NODES_OBJ, dtype=np.objects)

        # Set default values for node_type
        properties['int']['nodes']['node_type'] = NODE_TYPES['junction']
        # Set default values for point_type
        properties['int']['points']['point_type'] = POINT_TYPES['interior']

        i = 0 # nodes index

        for start_node in self.network_graph:
            downstream_nodes = self.network_graph[start_node]

            downstream_link_names = [
                link for end_node_name in downstream_nodes
                for link in self.network_graph[start_node][end_node_name]
            ]

            # Check if start junction is a reservoir
            if start_node in self.wn.reservoir_name_list:
                properties['int']['nodes']['node_type'][start_node] = NODE_TYPES['reservoir']

            # Define start junction demand
            H0_start = float(self.steady_state_sim.node['head'][start_node])
            fixed_demand = float(self.steady_state_sim.node['demand'][start_node])
            properties['float']['nodes']['fixed_demand'][start_node] = fixed_demand
            properties['float']['nodes']['demand_coeff'][start_node] = fixed_demand / (2*G*H0_start**0.5)

            # Update downstream nodes
            for j, end_node in enumerate(downstream_nodes):
                link_name = downstream_link_names[j]
                link = self.wn.get_link(link_name)


                # Check if end junction is a reservoir
                if end_node in self.wn.reservoir_name_list:
                    properties['int']['nodes']['node_type'][end_node] = NODE_TYPES['reservoir']

                # Define end junction demand
                H0_end = float(self.steady_state_sim.node['head'][end_node])
                fixed_demand = float(self.steady_state_sim.node['demand'][end_node])
                properties['float']['nodes']['fixed_demand'][end_node] = fixed_demand
                properties['float']['nodes']['demand_coeff'][end_node] = fixed_demand / (2*G*H0_end**0.5)

                if link.link_type == 'Pipe':

                    # Friction factor based on D-W equation
                    Q0 = float(self.steady_state_sim.link['flowrate'][link_name])
                    pipe_diameter = link.diameter
                    pipe_area = (np.pi * link.diameter ** 2 / 4)
                    pipe_length = link.length
                    ffactor = (abs(H0_start - H0_end)*2*G*pipe_diameter) / (pipe_length * (Q0 / pipe_area)**2)

                    #  Points are stored in order, such that the i-1 and the i+1
                    #  correspond to the upstream and downstream nodes of the
                    #  i-th node
                    for idx in range(self.segments[link_name]+1):
                        properties['int']['points']['subindex'][i] = idx
                        properties['int']['points']['link_id'][i] = k
                        if idx == 0:
                            properties['int']['points']['point_type'][i] = POINT_TYPES['boundary']
                            if isnan(properties['obj']['nodes'][start_node]):
                                properties['obj']['nodes'][start_node] = [i]
                            else:
                                properties['obj']['nodes'][start_node].append(i)
                        elif idx == self.segments[link_name]:
                            properties['int']['points']['point_type'][i] = POINT_TYPES['boundary']
                            if isnan(properties['obj']['nodes'][end_node]):
                                properties['obj']['nodes'][end_node] = [i]
                            else:
                                properties['obj']['nodes'][end_node].append(i)
                        dx = pipe_length / self.segments[link_name]
                        properties['float']['points']['B'][i] = self.wave_speeds[link_name] / (G*pipe_area)
                        properties['float']['points']['R'][i] = ffactor*dx / (2*G*pipe_diameter*pipe_area**2)
                        i += 1
                elif link.link_type == 'Valve':
                    properties['int']['nodes']['node_type'][start_node] = NODE_TYPES['valve']
                    properties['int']['nodes']['node_type'][end_node] = NODE_TYPES['valve']
                    properties['int']['valves']['upstream_node'][link_name] = properties['int']['nodes'].index.get_loc(start_node)
                    properties['int']['valves']['downstream_node'][link_name] = properties['int']['nodes'].index.get_loc(end_node)
                    properties['float']['valves']['area'][link_name] = (np.pi * link.diameter ** 2 / 4)
                elif link.link_type == 'Pump':
                    properties['int']['nodes']['node_type'][start_node] = NODE_TYPES['pump']
                    properties['int']['nodes']['node_type'][end_node] = NODE_TYPES['pump']
                    properties['int']['pumps']['upstream_junction'][link_name] = properties['int']['nodes'].index.get_loc(start_node)
                    properties['int']['pumps']['downstream_junction'][link_name] = properties['int']['nodes'].index.get_loc(end_node)
                    (a, b, c,) = link.get_head_curve_coefficients()
                    properties['float']['pumps']'a'][link_name] = a
                    properties['float']['pumps']'b'][link_name] = b
                    properties['float']['pumps']'c'][link_name] = c

        if self.num_points != (i-1):
            raise Exception("Internal error - incorrect size for DF")
        return properties