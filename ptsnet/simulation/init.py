import os
import numpy as np

from wntr.epanet.io import InpFile
from ptsnet.arrays import Table
from ptsnet.epanet.toolkit import ENepanet
from ptsnet.epanet.util import EN, FlowUnits, HydParam, to_si
from ptsnet.simulation.constants import G, TOL, FLOOR_FFACTOR, CEIL_FFACTOR, DEFAULT_FFACTOR
from ptsnet.simulation.constants import NODE_PROPERTIES, PIPE_PROPERTIES, PUMP_PROPERTIES, VALVE_PROPERTIES, OPEN_PROTECTION_PROPERTIES, CLOSED_PROTECTION_PROPERTIES
from ptsnet.simulation.validation import check_compatibility
from ptsnet.arrays.selectors import SelectorSet
from ptsnet.utils.data import imerge

from time import time

class Initializer:

    def __init__(self, inpfile, period = 0, skip_compatibility_check = False, warnings_on = True, _super = None):
        self.wn = get_water_network(inpfile)
        self.ng = self.wn.get_graph()
        self.ss = get_initial_conditions(inpfile, period = period, wn = self.wn)
        self.num_points = 0
        self.num_segments = 0
        self.where = None
        self._super = _super

        if not skip_compatibility_check:
            if warnings_on:
                t = time()
            try:
                check_compatibility(wn=self.wn, ss=self.ss)
            except Exception as e:
                if warnings_on:
                    print("Elapsed time (model check): ", time() - t, '[s]')
                raise e
            if warnings_on:
                print("Success - Compatible Model")
                print("Elapsed time (model check): ", time() - t, '[s]')

    def create_secondary_elements(self):
        # Open Protection Devices
        num_open_protections = len(self.ss['open_protection'])
        if num_open_protections > 0:
            open_protections = Table(OPEN_PROTECTION_PROPERTIES, num_open_protections)
            open_protection_labels = list(self.ss['open_protection'].keys())
            open_protections.assign_labels(open_protection_labels)
            for elem, props in self.ss['open_protection'].items():
                open_protections.node[elem] = props['node']
                open_protections.area[elem] = props['area']
            self.ss['open_protection'] = open_protections

        # Closed Protection Devices
        num_closed_protections = len(self.ss['closed_protection'])
        if num_closed_protections > 0:
            closed_protections = Table(CLOSED_PROTECTION_PROPERTIES, num_closed_protections)
            closed_protection_labels = list(self.ss['closed_protection'].keys())
            closed_protections.assign_labels(closed_protection_labels)
            for elem, props in self.ss['closed_protection'].items():
                closed_protections.node[elem] = props['node']
                closed_protections.area[elem] = props['area']
                closed_protections.height[elem] = props['height']
                closed_protections.water_level[elem] = props['water_level']
            self.ss['closed_protection'] = closed_protections

    def create_selectors(self):
        '''
        .points
            ['to_pipes']

            ['are_uboundaries']

            ['are_dboundaries']

            ['are_boundaries']

            ['start_valve']

            ['end_valve']

            ['single_valve']

            ['are_valve']

            ['start_pump']

            ['end_pump']

            ['single_pump']

            ['are_pump']

        .pipes
            ['to_nodes']

        .nodes
            ['not_in_pipes']

            ['in_pipes']

            ['to_points']

            ['to_points_are_uboundaries']

        .valves
            ['are_inline']

        '''
        self.where = SelectorSet(['points', 'pipes', 'nodes', 'valves', 'pumps', 'surge_protections'])

        self.where.pipes['to_nodes'] = imerge(self.ss['pipe'].start_node, self.ss['pipe'].end_node)
        pipes_idx = np.cumsum(self.ss['pipe'].segments+1).astype(int)
        self.where.points['to_pipes'] = np.zeros(pipes_idx[-1], dtype=int)
        for i in range(1, len(pipes_idx)):
            start = pipes_idx[i-1]
            end = pipes_idx[i]
            self.where.points['to_pipes'][start:end] = i
        self.where.points['are_uboundaries'] = np.cumsum(self.ss['pipe'].segments.astype(np.int)+1) - 1
        self.where.points['are_dboundaries'] = self.where.points['are_uboundaries'] - self.ss['pipe'].segments.astype(np.int)
        self.where.points['are_boundaries'] = imerge(self.where.points['are_dboundaries'], self.where.points['are_uboundaries'])
        self.where.points['are_boundaries',] = (np.arange(len(self.where.points['are_boundaries'])) % 2 != 0)

    def create_secondary_selectors(self):
        node_points_order = np.argsort(self.where.pipes['to_nodes'])
        self.where.nodes['not_in_pipes'] = np.isin(np.arange(self.wn.num_nodes),
            np.unique(np.concatenate((
                self.ss['valve'].start_node, self.ss['valve'].end_node,
                self.ss['pump'].start_node, self.ss['pump'].end_node)))
        )

        self.where.nodes['in_pipes'] = np.isin(self.where.pipes['to_nodes'], np.where(self.where.nodes['not_in_pipes'])[0])

        self.where.nodes['to_points'] = self.where.points['are_boundaries'][node_points_order]
        self.where.nodes['to_points_are_uboundaries'] = np.isin(self.where.nodes['to_points'], self.where.points['are_uboundaries'])
        self.where.nodes['to_points_are_dboundaries'] = np.isin(self.where.nodes['to_points'], self.where.points['are_dboundaries'])
        self.where.nodes['to_points',] = self.ss['node'].degree - self.where.nodes['not_in_pipes'] # Real degree

        # # Valve selectors
        self._create_nonpipe_selectors('valve')

        # # Pump selectors
        self._create_nonpipe_selectors('pump')

        # # Protection devices
        self._create_nonpipe_selectors('open_protection')
        self._create_nonpipe_selectors('closed_protection')

    def set_wave_speeds(self, default_wave_speed = 1000, wave_speed_file_path = None, delimiter = ',', wave_speed_method = 'optimal'):

            modified_lines = 0
            if wave_speed_file_path != None:
                with open(wave_speed_file_path, 'r') as f:
                    for line in f:
                        line = line.strip()
                        if len(line) <= 1:
                            raise ValueError("The wave_speed file has to have to entries per line 'pipe,wave_speed'")
                        pipe, wave_speed = line.split(delimiter)
                        self.ss['pipe'].wave_speed[pipe] = float(wave_speed)
                        modified_lines += 1

            if default_wave_speed != None:
                self.ss['pipe'].wave_speed[:] = default_wave_speed
                modified_lines = self.wn.num_pipes

            if modified_lines != self.wn.num_pipes:
                self.ss['pipe'].wave_speed[:] = 0
                excep = "The file does not specify wave speed values for all the pipes,\n"
                excep += "it is necessary to define a default wave speed value"
                raise ValueError(excep)

            self.ss['pipe'].desired_wave_speed = self.ss['pipe'].wave_speed
            self._set_segments(wave_speed_method)
            return True

    def _set_segments(self, wave_speed_method = 'optimal'):
        # method \in {'critical', 'user', 'dt', 'optimal'}
        if wave_speed_method in ('critical', 'dt', 'optimal'):
            self.ss['pipe'].segments = self.ss['pipe'].length
            self.ss['pipe'].segments /= self.ss['pipe'].wave_speed
            # Maximum time_step in the system to capture waves in all pipes
            max_dt = min(self.ss['pipe'].segments) / 2 # at least 2 segments in critical pipe
            self._super.settings.time_step = min(self._super.settings.time_step, max_dt)
            # The number of segments is defined
            self.ss['pipe'].segments /= self._super.settings.time_step
        elif wave_speed_method == 'user':
            self.ss['pipe'].segments = self.ss['pipe'].length
            self.ss['pipe'].segments /= (self.ss['pipe'].wave_speed * self._super.settings.time_step)
        elif wave_speed_method == 'dt':
            phi = self.ss['pipe'].length
        else:
            raise ValueError("Method is not compatible. Try: ['critical', 'user']")

        int_segments = np.round(self.ss['pipe'].segments)
        int_segments[int_segments < 2] = 2

        if wave_speed_method in ('critical', 'user'):
            # The wave_speed values are adjusted to compensate the truncation error
            self.ss['pipe'].wave_speed = self.ss['pipe'].wave_speed * self.ss['pipe'].segments/int_segments
            self.ss['pipe'].segments = int_segments
        elif wave_speed_method == 'dt':
            # The wave_speed is not adjusted when wave_speed_method == 'dt'
            # since the error is absorved by the time step
            self.ss['pipe'].segments = int_segments
        elif wave_speed_method == 'optimal':
            self.ss['pipe'].segments = int_segments
            phi = self.ss['pipe'].length / (self.ss['pipe'].wave_speed * self.ss['pipe'].segments)
            theta = np.dot(phi,np.ones_like(phi)) / np.dot(phi, phi)
            self._super.settings.time_step = 1/theta
            self.ss['pipe'].wave_speed = self.ss['pipe'].wave_speed * (phi*theta)
        self.ss['pipe'].dx = self.ss['pipe'].length / self.ss['pipe'].segments
        self.num_segments = int(sum(self.ss['pipe'].segments))
        self.num_points = self.num_segments + self.wn.num_pipes

    def _create_nonpipe_selectors(self, object_type):
        '''

        Notes:
        - Non-pipe selectors

        '''
        if object_type in ('valve', 'pump'):
            x1 = np.isin(self.where.pipes['to_nodes'], self.ss[object_type].start_node[self.ss[object_type].is_inline])
            x2 = np.isin(self.where.pipes['to_nodes'], self.ss[object_type].end_node[self.ss[object_type].is_inline])
            if object_type == 'valve':
                x3 = np.isin(self.where.pipes['to_nodes'], self.ss[object_type].start_node[~self.ss[object_type].is_inline])
            elif object_type == 'pump':
                x3 = np.isin(self.where.pipes['to_nodes'], self.ss[object_type].end_node[~self.ss[object_type].is_inline])
            self.where.points['start_' + object_type] = np.sort(self.where.points['are_boundaries'][x1])
            ordered = np.argsort(self.ss[object_type].start_node)
            ordered_end = np.argsort(self.ss[object_type].end_node)
            self.where.points['start_' + object_type,] = ordered[self.ss[object_type].is_inline[ordered]]
            last_order = np.argsort(self.where.points['start_' + object_type,])
            self.where.points['start_' + object_type][:] = self.where.points['start_' + object_type][last_order]
            self.where.points['start_' + object_type,][:] = self.where.points['start_' + object_type,][last_order]
            self.where.points['end_' + object_type] = np.sort(self.where.points['are_boundaries'][x2])
            self.where.points['end_' + object_type,] = ordered_end[self.ss[object_type].is_inline[ordered_end]]
            self.where.__dict__[object_type + 's']['are_inline'] = self.ss[object_type].is_inline
            self.where.__dict__[object_type + 's']['are_inline',] = last_order
            last_order = np.argsort(self.where.points['end_' + object_type,])
            self.where.points['end_' + object_type][:] = self.where.points['end_' + object_type][last_order]
            self.where.points['end_' + object_type,][:] = self.where.points['end_' + object_type,][last_order]
            self.where.points['single_' + object_type] = np.sort(self.where.points['are_boundaries'][x3])
            self.where.points['single_' + object_type,] = ordered[~self.ss[object_type].is_inline[ordered]]
            last_order = np.argsort(self.where.points['single_' + object_type,])
            self.where.points['single_' + object_type][:] = self.where.points['single_' + object_type][last_order]
            self.where.points['single_' + object_type,][:] = self.where.points['single_' + object_type,][last_order]
            self.where.points['are_' + object_type] = np.concatenate((
            self.where.points['single_' + object_type],
            self.where.points['start_' + object_type],
            self.where.points['end_' + object_type]))
            self.where.points['are_' + object_type].sort()
        elif object_type in ('open_protection', 'closed_protection'):
            protection_type = object_type[:object_type.find('_')]
            if self.ss[f'{protection_type}_protection']:
                self.where.nodes[f'are_{protection_type}_protection'] = self.ss[f'{protection_type}_protection'].node
                node_end_idx = np.cumsum(self.where.nodes['to_points',])[self.ss[f'{protection_type}_protection'].node] - 1
                node_start_idx = node_end_idx - self.where.nodes['to_points',][self.ss[f'{protection_type}_protection'].node] + 1
                self.where.points[f'start_{protection_type}_protection'] = self.where.nodes['to_points'][node_start_idx]
                self.where.points[f'end_{protection_type}_protection'] = self.where.nodes['to_points'][node_end_idx]
                self.where.points[f'are_{protection_type}_protection'] = imerge(self.where.points[f'start_{protection_type}_protection'], self.where.points[f'end_{protection_type}_protection'])

def get_water_network(inpfile):
    ENFile = InpFile()
    return ENFile.read(inpfile)

def get_initial_conditions(inpfile, period = 0, wn = None):

    # EPANET initialization

    file_prefix, _ = os.path.splitext(inpfile)
    rptfile = file_prefix + '.rpt'
    outfile = file_prefix + '.bin'

    if wn is None:
        wn = get_water_network(inpfile)

    network_graph = wn.get_graph()
    EPANET = ENepanet()
    EPANET.ENopen(inpfile, rptfile, outfile)
    EPANET.ENopenH()
    EPANET.ENinitH(0)

    # Data structures for node and link initial conditions
    nodes = Table(NODE_PROPERTIES, wn.num_nodes)
    node_labels = []
    pipes = Table(PIPE_PROPERTIES, wn.num_pipes)
    pipe_labels = []
    valves = Table(VALVE_PROPERTIES, wn.num_valves)
    valve_labels = []
    pumps = Table(PUMP_PROPERTIES, wn.num_pumps)
    pump_labels = []

    ss = {
        'node' : nodes,
        'pipe' : pipes,
        'valve' : valves,
        'pump' : pumps,
        'open_protection' : {},
        'closed_protection' : {}
    }

    # Run EPANET simulation
    t = 0
    while EPANET.ENnextH() > 0 and t <= period: # EPS
        EPANET.ENrunH()
        t += 1
    if t == 0: # Not EPS
        EPANET.ENrunH()

    flow_units = FlowUnits(EPANET.ENgetflowunits())

    # Retrieve node conditions
    for i in range(1, wn.num_nodes+1):
        node_id = EPANET.ENgetnodeid(i)
        node_labels.append(node_id)
        ss['node'].leak_coefficient[i-1] = EPANET.ENgetnodevalue(i, EN.EMITTER)
        ss['node'].demand[i-1] = EPANET.ENgetnodevalue(i, EN.DEMAND)
        ss['node'].head[i-1] = EPANET.ENgetnodevalue(i, EN.HEAD)
        ss['node'].pressure[i-1] = EPANET.ENgetnodevalue(i, EN.PRESSURE)
        ss['node'].type[i-1] = EPANET.ENgetnodetype(i)
        z = EPANET.ENgetnodevalue(i, EN.ELEVATION)
        if ss['node'].type[i-1] == EN.RESERVOIR:
            z = 0
        elif ss['node'].type[i-1] == EN.TANK:
            z = ss['node'].head[i-1] - ss['node'].pressure[i-1]
        ss['node'].elevation[i-1] = z
        ss['node'].degree[i-1] = network_graph.degree(node_id)

    # Unit conversion
    to_si(flow_units, ss['node'].leak_coefficient, HydParam.EmitterCoeff)
    to_si(flow_units, ss['node'].demand, HydParam.Flow)
    to_si(flow_units, ss['node'].head, HydParam.HydraulicHead)
    to_si(flow_units, ss['node'].pressure, HydParam.Pressure)
    to_si(flow_units, ss['node'].elevation, HydParam.Elevation)

    non_pipe_nodes = []
    p, pp, v = 0, 0, 0 # pipes, pumps, valves
    for i in range(1, wn.num_links+1):

        link = wn.get_link(EPANET.ENgetlinkid(i))
        ltype = link.link_type.lower()

        if link.link_type == 'Pipe':
            k = p; p += 1
        elif link.link_type == 'Pump':
            k = pp; pp += 1
        elif link.link_type == 'Valve':
            k = v; v += 1

        ss[ltype].start_node[k], ss[ltype].end_node[k] = EPANET.ENgetlinknodes(i)
        ss[ltype].flowrate[k] = EPANET.ENgetlinkvalue(i, EN.FLOW)
        ss[ltype].velocity[k] = EPANET.ENgetlinkvalue(i, EN.VELOCITY)

        # Indexes are adjusted to fit the new Table / Indexing in EPANET's C code starts in 1
        ss[ltype].start_node[k] -= 1
        ss[ltype].end_node[k] -= 1

        flow = to_si(flow_units, [ss[ltype].flowrate[k]], HydParam.Flow)[0]
        if abs(flow) < TOL:
            ss[ltype].direction[k] = 0
            ss[ltype].flowrate[k] = 0
            if link.link_type == 'Pipe':
                ss[ltype].ffactor[k] = DEFAULT_FFACTOR
        elif flow > TOL:
            ss[ltype].direction[k] = 1
        else:
            ss[ltype].direction[k] = -1
            ss[ltype].flowrate[k] *= -1
            ss[ltype].start_node[k], ss[ltype].end_node[k] = ss[ltype].end_node[k], ss[ltype].start_node[k]

        if ss['node'].degree[ss[ltype].start_node[k]] >= 2 and \
            ss['node'].degree[ss[ltype].end_node[k]] >= 2:
            ss[ltype].is_inline[k] = True

        if link.link_type in ('Pipe', 'Valve'):
            ss[ltype].diameter[k] = link.diameter
            ss[ltype].area[k] = np.pi * link.diameter ** 2 / 4
            ss[ltype].type[k] = EPANET.ENgetlinktype(i)
            ss[ltype].head_loss[k] = EPANET.ENgetlinkvalue(i, EN.HEADLOSS)

        if link.link_type == 'Pipe':
            pipe_labels.append(link.name)
            ss[ltype].length[k] = link.length
        elif link.link_type == 'Pump':
            pump_labels.append(link.name)
            ss[ltype].initial_status[k] = link.initial_status
            ss[ltype].setting[k] = ss[ltype].initial_status[k]
            ss[ltype].head_loss[k] = EPANET.ENgetlinkvalue(i, EN.HEADLOSS)
            # Pump curve parameters
            qp, hp = list(zip(*link.get_pump_curve().points)); qp = list(qp); hp = list(hp)
            qpp = to_si(flow_units, float(ss[ltype].flowrate[k]), HydParam.Flow)
            hpp = to_si(flow_units, float(ss[ltype].head_loss[k]), HydParam.HydraulicHead)
            qp.pop(); hp.pop(); qp.append(qpp); hp.append(abs(hpp))
            order = np.argsort(qp)
            qp = np.array(qp)[order]
            hp = np.array(hp)[order]
            ss[ltype].a2[k], ss[ltype].a1[k], ss[ltype].Hs[k] = np.polyfit(qp, hp, 2)
            # Source head
            ss[ltype].source_head[k] = ss['node'].head[ss[ltype].start_node[k]]
            non_pipe_nodes += [ss[ltype].start_node[k], ss[ltype].end_node[k]]
        elif link.link_type == 'Valve':
            valve_labels.append(link.name)
            ss[ltype].initial_status[k] = EPANET.ENgetlinkvalue(i, EN.INITSTATUS)
            ss[ltype].setting[k] = ss[ltype].initial_status[k]
            ss[ltype].flowrate[k] = to_si(flow_units, float(ss[ltype].flowrate[k]), HydParam.Flow)
            ha = ss['node'].head[ss[ltype].start_node[k]]
            hb = ss['node'].head[ss[ltype].end_node[k]] if ss['node'].degree[ss[ltype].end_node[k]] > 1 else 0
            hl = ha - hb
            if hl > 0:
                ss[ltype].K[k] = ss[ltype].flowrate[k]/(ss[ltype].area[k]*(2*G*hl)**0.5)
            non_pipe_nodes += [ss[ltype].start_node[k], ss[ltype].end_node[k]]

    EPANET.ENcloseH()
    EPANET.ENclose()

    # Unit conversion
    to_si(flow_units, ss['pipe'].head_loss, HydParam.HydraulicHead)
    to_si(flow_units, ss['pipe'].flowrate, HydParam.Flow)
    to_si(flow_units, ss['pump'].flowrate, HydParam.Flow)
    to_si(flow_units, ss['pipe'].velocity, HydParam.Velocity)
    to_si(flow_units, ss['pump'].velocity, HydParam.Velocity)
    to_si(flow_units, ss['pump'].head_loss, HydParam.HydraulicHead)
    to_si(flow_units, ss['valve'].head_loss, HydParam.HydraulicHead)
    to_si(flow_units, ss['valve'].velocity, HydParam.Velocity)

    idx = ss['pipe'].ffactor == 0
    ss['pipe'].ffactor[idx] = \
        (2*G*ss['pipe'].diameter[idx] * ss['pipe'].head_loss[idx]) \
            / (ss['pipe'].length[idx] * ss['pipe'].velocity[idx]**2)

    # ss['pipe'].ffactor[ss['pipe'].ffactor >= CEIL_FFACTOR] = CEIL_FFACTOR
    # ss['pipe'].ffactor[ss['pipe'].ffactor <= FLOOR_FFACTOR] = FLOOR_FFACTOR

    ss['valve'].curve_index.fill(-1)
    ss['pump'].curve_index.fill(-1)

    demanded = np.logical_and(ss['node'].type != EN.RESERVOIR, ss['node'].type != EN.TANK)
    KeKd = ss['node'].demand[demanded] / np.sqrt(ss['node'].pressure[demanded])
    ss['node'].demand_coefficient[demanded] = KeKd - ss['node'].leak_coefficient[demanded]

    nodes.assign_labels(node_labels)
    pipes.assign_labels(pipe_labels)
    valves.assign_labels(valve_labels)
    pumps.assign_labels(pump_labels)

    non_pipe_nodes = np.array(non_pipe_nodes)
    zero_flow_pipes = ss['pipe'].flowrate == 0
    zf = ss['pipe'].start_node[zero_flow_pipes]
    zf = np.concatenate((zf, ss['pipe'].end_node[zero_flow_pipes]))
    _fix_zero_flow_convention('valve', non_pipe_nodes, zf, wn, ss)
    _fix_zero_flow_convention('pump', non_pipe_nodes, zf, wn, ss)

    return ss

def _fix_zero_flow_convention(ltype, non_pipe_nodes, zf, wn, ss):
    # Define flow convention for zero flow pipes attached to
    zero_flow = np.where(np.isin(ss[ltype].start_node, zf))[0]
    for k in zero_flow:
        upipe = wn.get_links_for_node(ss['node'].ilabel(ss[ltype].start_node[k]))
        upipe.remove(ss[ltype].labels[k])
        upipe = ss['pipe'].lloc(upipe[0])
        dpipe = wn.get_links_for_node(ss['node'].ilabel(ss[ltype].end_node[k]))
        dpipe.remove(ss[ltype].labels[k])
        dpipe = ss['pipe'].lloc(dpipe[0])

        if ss['pipe'].end_node[upipe] != ss[ltype].start_node[k]:
            ss['pipe'].direction[upipe] = -1 if ss['pipe'].direction[upipe] != -1 else 1
            ss['pipe'].start_node[upipe], ss['pipe'].end_node[upipe] = ss['pipe'].end_node[upipe], ss['pipe'].start_node[upipe]
        else:
            if ss['pipe'].direction[upipe] == 0:
                ss['pipe'].direction[upipe] = 1

        if ss['pipe'].start_node[dpipe] != ss[ltype].end_node[k]:
            ss['pipe'].direction[dpipe] = -1 if ss['pipe'].direction[dpipe] != -1 else 1
            ss['pipe'].start_node[dpipe], ss['pipe'].end_node[dpipe] = ss['pipe'].end_node[dpipe], ss['pipe'].start_node[dpipe]
        else:
            if ss['pipe'].direction[dpipe] == 0:
                ss['pipe'].direction[dpipe] = 1