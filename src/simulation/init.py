import os
import numpy as np

from wntr.epanet.io import InpFile
from ptsnet.arrays import Table
from ptsnet.epanet.toolkit import ENepanet
from ptsnet.epanet.util import EN, FlowUnits, HydParam, to_si
from ptsnet.simulation.constants import G, TOL, FLOOR_FFACTOR, CEIL_FFACTOR, DEFAULT_FFACTOR
from ptsnet.simulation.constants import NODE_PROPERTIES, PIPE_PROPERTIES, PUMP_PROPERTIES, VALVE_PROPERTIES
from ptsnet.simulation.validation import check_compatibility
from ptsnet.arrays.selectors import SelectorSet
from ptsnet.utils.data import imerge

from time import time

class Initializator:

    def __init__(self, inpfile, period = 0, skip_compatibility_check = False, warnings_on = True, _super = None):
        self.wn = get_water_network(inpfile)
        self.ng = self.wn.get_graph()
        self.ic = get_initial_conditions(inpfile, period = period, wn = self.wn)
        self.num_points = 0
        self.num_segments = 0
        self.where = None
        self._super = _super

        if not skip_compatibility_check:
            if warnings_on:
                t = time()
            try:
                check_compatibility(wn=self.wn, ic=self.ic)
            except Exception as e:
                if warnings_on:
                    print("Elapsed time (model check): ", time() - t, '[s]')
                raise e
            if warnings_on:
                print("Success - Compatible Model")
                print("Elapsed time (model check): ", time() - t, '[s]')

    def set_wave_speeds(self, default_wave_speed = None, wave_speed_file = None, delimiter = ',', wave_speed_method = 'critical'):
            if default_wave_speed is None and wave_speed_file is None:
                raise ValueError("wave_speed was not specified")

            if not default_wave_speed is None:
                self.ic['pipe'].wave_speed[:] = default_wave_speed

            modified_lines = 0
            if not wave_speed_file is None:
                with open(wave_speed_file, 'r') as f:
                    for line in f:
                        line = line.strip()
                        if len(line) <= 1:
                            raise ValueError("The wave_speed file has to have to entries per line 'pipe,wave_speed'")
                        pipe, wave_speed = line.split(delimiter)
                        self.ic['pipe'].wave_speed[pipe] = float(wave_speed)
                        modified_lines += 1
            else:
                self._set_segments(wave_speed_method)
                return True

            if modified_lines != self.wn.num_pipes:
                self.ic['pipe'].wave_speed[:] = 0
                excep = "The file does not specify wave speed values for all the pipes,\n"
                excep += "it is necessary to define a default wave speed value"
                raise ValueError(excep)

            self._set_segments(wave_speed_method)
            return True

    def _set_segments(self, wave_speed_method = 'optimal'):
        # method \in {'critical', 'user', 'dt', 'optimal'}
        if wave_speed_method in ('critical', 'dt', 'optimal'):
            self.ic['pipe'].segments = self.ic['pipe'].length
            self.ic['pipe'].segments /= self.ic['pipe'].wave_speed
            # Maximum time_step in the system to capture waves in all pipes
            max_dt = min(self.ic['pipe'].segments) / 2 # at least 2 segments in critical pipe
            self._super.settings.time_step = min(self._super.settings.time_step, max_dt)
            # The number of segments is defined
            self.ic['pipe'].segments /= self._super.settings.time_step
        elif wave_speed_method == 'user':
            self.ic['pipe'].segments = self.ic['pipe'].length
            self.ic['pipe'].segments /= (self.ic['pipe'].wave_speed * self._super.settings.time_step)
        elif wave_speed_method == 'dt':
            phi = self.ic['pipe'].length
        else:
            raise ValueError("Method is not compatible. Try: ['critical', 'user']")

        int_segments = np.round(self.ic['pipe'].segments)
        int_segments[int_segments < 2] = 2

        if wave_speed_method in ('critical', 'user'):
            # The wave_speed values are adjusted to compensate the truncation error
            self.ic['pipe'].wave_speed = self.ic['pipe'].wave_speed * self.ic['pipe'].segments/int_segments
            self.ic['pipe'].segments = int_segments
        elif wave_speed_method == 'dt':
            # The wave_speed is not adjusted when wave_speed_method == 'dt'
            # since the error is absorved by the time step
            self.ic['pipe'].segments = int_segments
        elif wave_speed_method == 'optimal':
            self.ic['pipe'].segments = int_segments
            phi = self.ic['pipe'].length / (self.ic['pipe'].wave_speed * self.ic['pipe'].segments)
            theta = np.dot(phi,np.ones_like(phi)) / np.dot(phi, phi)
            self._super.settings.time_step = 1/theta
            self.ic['pipe'].wave_speed = self.ic['pipe'].wave_speed * (phi*theta)
        self.ic['pipe'].dx = self.ic['pipe'].length / self.ic['pipe'].segments
        self.num_segments = int(sum(self.ic['pipe'].segments))
        self.num_points = self.num_segments + self.wn.num_pipes

    def _create_nonpipe_selectors(self, object_type):
        '''

        Notes:

        - Non-pipe selectors
        '''

        x1 = np.isin(self.where.pipes['to_nodes'], self.ic[object_type].start_node[self.ic[object_type].is_inline])
        x2 = np.isin(self.where.pipes['to_nodes'], self.ic[object_type].end_node[self.ic[object_type].is_inline])
        if object_type == 'valve':
            x3 = np.isin(self.where.pipes['to_nodes'], self.ic[object_type].start_node[~self.ic[object_type].is_inline])
        elif object_type == 'pump':
            x3 = np.isin(self.where.pipes['to_nodes'], self.ic[object_type].end_node[~self.ic[object_type].is_inline])
        self.where.points['start_inline_' + object_type] = np.sort(self.where.points['are_boundaries'][x1])
        ordered = np.argsort(self.ic[object_type].start_node)
        ordered_end = np.argsort(self.ic[object_type].end_node)
        self.where.points['start_inline_' + object_type,] = ordered[self.ic[object_type].is_inline[ordered]]
        last_order = np.argsort(self.where.points['start_inline_' + object_type,])
        self.where.points['start_inline_' + object_type][:] = self.where.points['start_inline_' + object_type][last_order]
        self.where.points['start_inline_' + object_type,][:] = self.where.points['start_inline_' + object_type,][last_order]
        self.where.points['end_inline_' + object_type] = np.sort(self.where.points['are_boundaries'][x2])
        self.where.points['end_inline_' + object_type,] = ordered_end[self.ic[object_type].is_inline[ordered_end]]
        self.where.__dict__[object_type + 's']['are_inline'] = self.ic[object_type].is_inline
        self.where.__dict__[object_type + 's']['are_inline',] = last_order
        last_order = np.argsort(self.where.points['end_inline_' + object_type,])
        self.where.points['end_inline_' + object_type][:] = self.where.points['end_inline_' + object_type][last_order]
        self.where.points['end_inline_' + object_type,][:] = self.where.points['end_inline_' + object_type,][last_order]
        self.where.points['are_single_' + object_type] = np.sort(self.where.points['are_boundaries'][x3])
        self.where.points['are_single_' + object_type,] = ordered[~self.ic[object_type].is_inline[ordered]]
        last_order = np.argsort(self.where.points['are_single_' + object_type,])
        self.where.points['are_single_' + object_type][:] = self.where.points['are_single_' + object_type][last_order]
        self.where.points['are_single_' + object_type,][:] = self.where.points['are_single_' + object_type,][last_order]
        self.where.points['are_' + object_type] = np.concatenate((
                self.where.points['are_single_' + object_type],
                self.where.points['start_inline_' + object_type],
                self.where.points['end_inline_' + object_type]))
        self.where.points['are_' + object_type].sort()

    def create_selectors(self):
        '''
        .points
            ['to_pipes']

            ['are_uboundaries']

            ['are_dboundaries']

            ['are_boundaries']

            ['start_inline_valve']

            ['end_inline_valve']

            ['are_single_valve']

            ['are_valve']

            ['start_inline_pump']

            ['end_inline_pump']

            ['are_single_pump']

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
        self.where = SelectorSet(['points', 'pipes', 'nodes', 'valves', 'pumps'])

        self.where.pipes['to_nodes'] = imerge(self.ic['pipe'].start_node, self.ic['pipe'].end_node)
        pipes_idx = np.cumsum(self.ic['pipe'].segments+1).astype(int)
        self.where.points['to_pipes'] = np.zeros(pipes_idx[-1], dtype=int)
        for i in range(1, len(pipes_idx)):
            start = pipes_idx[i-1]
            end = pipes_idx[i]
            self.where.points['to_pipes'][start:end] = i
        self.where.points['are_uboundaries'] = np.cumsum(self.ic['pipe'].segments.astype(np.int)+1) - 1
        self.where.points['are_dboundaries'] = self.where.points['are_uboundaries'] - self.ic['pipe'].segments.astype(np.int)
        self.where.points['are_boundaries'] = imerge(self.where.points['are_dboundaries'], self.where.points['are_uboundaries'])
        self.where.points['are_boundaries',] = (np.arange(len(self.where.points['are_boundaries'])) % 2 != 0)

        node_points_order = np.argsort(self.where.pipes['to_nodes'])
        self.where.nodes['not_in_pipes'] = np.isin(np.arange(self.wn.num_nodes),
            np.unique(np.concatenate((
                self.ic['valve'].start_node, self.ic['valve'].end_node,
                self.ic['pump'].start_node, self.ic['pump'].end_node)))
        )

        self.where.nodes['in_pipes'] = np.isin(self.where.pipes['to_nodes'], np.where(self.where.nodes['not_in_pipes'])[0])

        self.where.nodes['to_points'] = self.where.points['are_boundaries'][node_points_order]
        self.where.nodes['to_points_are_uboundaries'] = (np.arange(2*self.wn.num_pipes) % 2 != 0).astype(int)[node_points_order]
        self.where.nodes['to_points',] = self.ic['node'].degree - self.where.nodes['not_in_pipes'] # Real degree

        # # Valve selectors
        self._create_nonpipe_selectors('valve')

        # # Pump selectors
        self._create_nonpipe_selectors('pump')

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

    ic = {
        'node' : nodes,
        'pipe' : pipes,
        'valve' : valves,
        'pump' : pumps,
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
        ic['node'].leak_coefficient[i-1] = EPANET.ENgetnodevalue(i, EN.EMITTER)
        ic['node'].demand[i-1] = EPANET.ENgetnodevalue(i, EN.DEMAND)
        ic['node'].head[i-1] = EPANET.ENgetnodevalue(i, EN.HEAD)
        ic['node'].pressure[i-1] = EPANET.ENgetnodevalue(i, EN.PRESSURE)
        ic['node'].type[i-1] = EPANET.ENgetnodetype(i)
        z = EPANET.ENgetnodevalue(i, EN.ELEVATION)
        if ic['node'].type[i-1] == EN.RESERVOIR:
            z = 0
        elif ic['node'].type[i-1] == EN.TANK:
            z = ic['node'].head[i-1] - ic['node'].pressure[i-1]
        ic['node'].elevation[i-1] = z
        ic['node'].degree[i-1] = network_graph.degree(node_id)

    # Unit conversion
    to_si(flow_units, ic['node'].leak_coefficient, HydParam.EmitterCoeff)
    to_si(flow_units, ic['node'].demand, HydParam.Flow)
    to_si(flow_units, ic['node'].head, HydParam.HydraulicHead)
    to_si(flow_units, ic['node'].pressure, HydParam.Pressure)
    to_si(flow_units, ic['node'].elevation, HydParam.Elevation)

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

        ic[ltype].start_node[k], ic[ltype].end_node[k] = EPANET.ENgetlinknodes(i)
        ic[ltype].flowrate[k] = EPANET.ENgetlinkvalue(i, EN.FLOW)
        ic[ltype].velocity[k] = EPANET.ENgetlinkvalue(i, EN.VELOCITY)

        # Indexes are adjusted to fit the new Table / Indexing in EPANET's C code starts in 1
        ic[ltype].start_node[k] -= 1
        ic[ltype].end_node[k] -= 1

        flow = to_si(flow_units, [ic[ltype].flowrate[k]], HydParam.Flow)[0]
        if abs(flow) < TOL:
            ic[ltype].direction[k] = 0
            ic[ltype].flowrate[k] = 0
            if link.link_type == 'Pipe':
                ic[ltype].ffactor[k] = DEFAULT_FFACTOR
        elif flow > TOL:
            ic[ltype].direction[k] = 1
        else:
            ic[ltype].direction[k] = -1
            ic[ltype].flowrate[k] *= -1
            ic[ltype].start_node[k], ic[ltype].end_node[k] = ic[ltype].end_node[k], ic[ltype].start_node[k]

        if ic['node'].degree[ic[ltype].start_node[k]] >= 2 and \
            ic['node'].degree[ic[ltype].end_node[k]] >= 2:
            ic[ltype].is_inline[k] = True

        if link.link_type in ('Pipe', 'Valve'):
            ic[ltype].diameter[k] = link.diameter
            ic[ltype].area[k] = np.pi * link.diameter ** 2 / 4
            ic[ltype].type[k] = EPANET.ENgetlinktype(i)
            ic[ltype].head_loss[k] = EPANET.ENgetlinkvalue(i, EN.HEADLOSS)

        if link.link_type == 'Pipe':
            pipe_labels.append(link.name)
            ic[ltype].length[k] = link.length
        elif link.link_type == 'Pump':
            pump_labels.append(link.name)
            ic[ltype].initial_status[k] = link.initial_status
            ic[ltype].setting[k] = ic[ltype].initial_status[k]
            ic[ltype].head_loss[k] = EPANET.ENgetlinkvalue(i, EN.HEADLOSS)
            # Pump curve parameters
            qp, hp = list(zip(*link.get_pump_curve().points)); qp = list(qp); hp = list(hp)
            qpp = to_si(flow_units, float(ic[ltype].flowrate[k]), HydParam.Flow)
            hpp = to_si(flow_units, float(ic[ltype].head_loss[k]), HydParam.HydraulicHead)
            qp.pop(); hp.pop(); qp.append(qpp); hp.append(abs(hpp))
            order = np.argsort(qp)
            qp = np.array(qp)[order]
            hp = np.array(hp)[order]
            ic[ltype].a2[k], ic[ltype].a1[k], ic[ltype].Hs[k] = np.polyfit(qp, hp, 2)
            # Source head
            ic[ltype].source_head[k] = ic['node'].head[ic[ltype].start_node[k]]
            non_pipe_nodes += [ic[ltype].start_node[k], ic[ltype].end_node[k]]
        elif link.link_type == 'Valve':
            valve_labels.append(link.name)
            ic[ltype].initial_status[k] = EPANET.ENgetlinkvalue(i, EN.INITSTATUS)
            ic[ltype].setting[k] = ic[ltype].initial_status[k]
            ic[ltype].flowrate[k] = to_si(flow_units, float(ic[ltype].flowrate[k]), HydParam.Flow)
            ha = ic['node'].head[ic[ltype].start_node[k]]
            hb = ic['node'].head[ic[ltype].end_node[k]] if ic['node'].degree[ic[ltype].end_node[k]] > 1 else 0
            hl = ha - hb
            if hl > 0:
                ic[ltype].K[k] = ic[ltype].flowrate[k]/(ic[ltype].area[k]*(2*G*hl)**0.5)
            non_pipe_nodes += [ic[ltype].start_node[k], ic[ltype].end_node[k]]

    EPANET.ENcloseH()
    EPANET.ENclose()

    # Unit conversion
    to_si(flow_units, ic['pipe'].head_loss, HydParam.HydraulicHead)
    to_si(flow_units, ic['pipe'].flowrate, HydParam.Flow)
    to_si(flow_units, ic['pump'].flowrate, HydParam.Flow)
    to_si(flow_units, ic['pipe'].velocity, HydParam.Velocity)
    to_si(flow_units, ic['pump'].velocity, HydParam.Velocity)
    to_si(flow_units, ic['pump'].head_loss, HydParam.HydraulicHead)
    to_si(flow_units, ic['valve'].head_loss, HydParam.HydraulicHead)
    to_si(flow_units, ic['valve'].velocity, HydParam.Velocity)

    idx = ic['pipe'].ffactor == 0
    ic['pipe'].ffactor[idx] = \
        (2*G*ic['pipe'].diameter[idx] * ic['pipe'].head_loss[idx]) \
            / (ic['pipe'].length[idx] * ic['pipe'].velocity[idx]**2)

    # ic['pipe'].ffactor[ic['pipe'].ffactor >= CEIL_FFACTOR] = CEIL_FFACTOR
    # ic['pipe'].ffactor[ic['pipe'].ffactor <= FLOOR_FFACTOR] = FLOOR_FFACTOR

    ic['valve'].curve_index.fill(-1)
    ic['pump'].curve_index.fill(-1)

    demanded = np.logical_and(ic['node'].type != EN.RESERVOIR, ic['node'].type != EN.TANK)
    KeKd = ic['node'].demand[demanded] / np.sqrt(ic['node'].pressure[demanded])
    ic['node'].demand_coefficient[demanded] = KeKd - ic['node'].leak_coefficient[demanded]

    nodes.assign_labels(node_labels)
    pipes.assign_labels(pipe_labels)
    valves.assign_labels(valve_labels)
    pumps.assign_labels(pump_labels)

    non_pipe_nodes = np.array(non_pipe_nodes)
    zero_flow_pipes = ic['pipe'].flowrate == 0
    zf = ic['pipe'].start_node[zero_flow_pipes]
    zf = np.concatenate((zf, ic['pipe'].end_node[zero_flow_pipes]))
    _fix_zero_flow_convention('valve', non_pipe_nodes, zf, wn, ic)
    _fix_zero_flow_convention('pump', non_pipe_nodes, zf, wn, ic)

    return ic

def _fix_zero_flow_convention(ltype, non_pipe_nodes, zf, wn, ic):
    # Define flow convention for zero flow pipes attached to
    zero_flow = np.where(np.isin(ic[ltype].start_node, zf))[0]
    for k in zero_flow:
        upipe = wn.get_links_for_node(ic['node'].ilabel(ic[ltype].start_node[k]))
        upipe.remove(ic[ltype].labels[k])
        upipe = ic['pipe'].lloc(upipe[0])
        dpipe = wn.get_links_for_node(ic['node'].ilabel(ic[ltype].end_node[k]))
        dpipe.remove(ic[ltype].labels[k])
        dpipe = ic['pipe'].lloc(dpipe[0])

        if ic['pipe'].end_node[upipe] != ic[ltype].start_node[k]:
            ic['pipe'].direction[upipe] = -1 if ic['pipe'].direction[upipe] != -1 else 1
            ic['pipe'].start_node[upipe], ic['pipe'].end_node[upipe] = ic['pipe'].end_node[upipe], ic['pipe'].start_node[upipe]
        else:
            if ic['pipe'].direction[upipe] == 0:
                ic['pipe'].direction[upipe] = 1

        if ic['pipe'].start_node[dpipe] != ic[ltype].end_node[k]:
            ic['pipe'].direction[dpipe] = -1 if ic['pipe'].direction[dpipe] != -1 else 1
            ic['pipe'].start_node[dpipe], ic['pipe'].end_node[dpipe] = ic['pipe'].end_node[dpipe], ic['pipe'].start_node[dpipe]
        else:
            if ic['pipe'].direction[dpipe] == 0:
                ic['pipe'].direction[dpipe] = 1