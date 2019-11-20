import numpy as np

from collections import defaultdict as ddict
from phammer.simulation.init import Initializator
from phammer.arrays.arrays import Table2D, Table, ObjArray
from phammer.parallel.partitioning import even, get_partition
from phammer.simulation.constants import MEM_POOL_POINTS, PIPE_START_RESULTS, PIPE_END_RESULTS, NODE_RESULTS, POINT_PROPERTIES, G, COEFF_TOL
from phammer.simulation.util import is_iterable
from phammer.arrays.selectors import SelectorSet
from phammer.simulation.funcs import run_boundary_step, run_interior_step, run_pump_step, run_valve_step

class Worker:
    def __init__(self, **kwargs):
        self.send_queue = None
        self.recv_queue = None
        self.global_comm = kwargs['comm']
        self.comm = None
        self.rank = kwargs['rank']
        self.num_points = kwargs['num_points']
        self.num_processors = kwargs['num_processors']
        self.wn = kwargs['wn']
        self.ic = kwargs['ic']
        self.global_where = kwargs['where']
        self.time_steps = kwargs['time_steps']
        self.mem_pool_points = None
        self.point_properties = None
        self.pipe_start_results = None
        self.pipe_end_results = None
        self.node_results = None
        self.where = SelectorSet(['points', 'pipes', 'nodes', 'valves', 'pumps'])
        self.processors = even(self.num_points, self.num_processors)
        self.partition = get_partition(self.processors, self.rank, self.global_where, self.ic, self.wn)
        self._create_selectors()
        self._define_worker_comm_queues()
        self._define_dist_graph_comm()
        self._comm_buffer_head = []
        self._recv_points = []
        for r in self.recv_queue.values:
            self._comm_buffer_head.append(np.zeros(len(r)))
            self._recv_points.extend(r)
        self._comm_buffer_flow = np.array(self._comm_buffer_head)
        self._comm_buffer_head = np.array(self._comm_buffer_head)
        self._allocate_memory()
        self._load_initial_conditions()

    def _allocate_memory(self):
        self.mem_pool_points = Table2D(MEM_POOL_POINTS, len(self.partition['points']['global_idx']), 2)
        self.point_properties = Table(POINT_PROPERTIES, len(self.partition['points']['global_idx']))

        nodes = []
        nodes += list(self.partition['nodes']['global_idx'])
        nodes += list(self.partition['tanks']['global_idx'])
        nodes += list(self.partition['reservoirs']['global_idx'])
        nodes += list(self.ic['valve'].start_node[self.partition['inline_valves']['global_idx']])
        nodes += list(self.ic['valve'].end_node[self.partition['inline_valves']['global_idx']])
        nodes += list(self.ic['pump'].start_node[self.partition['inline_pumps']['global_idx']])
        nodes += list(self.ic['pump'].end_node[self.partition['inline_pumps']['global_idx']])
        nodes += list(self.ic['valve'].start_node[self.partition['single_valves']['global_idx']])
        nodes += list(self.ic['valve'].end_node[self.partition['single_valves']['global_idx']])
        nodes += list(self.ic['pump'].start_node[self.partition['single_pumps']['global_idx']])
        nodes += list(self.ic['pump'].end_node[self.partition['single_pumps']['global_idx']])
        nodes = np.unique(nodes)

        self.node_results = None
        if len(nodes) > 0:
            self.node_results = Table2D(NODE_RESULTS, len(nodes), self.time_steps, index = self.ic['node']._index_keys[nodes])
        points = self.partition['points']['global_idx']
        ppoints_start = points[self.where.points['are_dboundaries']]
        ppoints_end = points[self.where.points['are_uboundaries']]
        pipes_start = self.global_where.points['to_pipes'][ppoints_start]
        pipes_end = self.global_where.points['to_pipes'][ppoints_end]

        self.pipe_start_results = Table2D(PIPE_START_RESULTS, len(ppoints_start), self.time_steps, index = self.ic['pipe']._index_keys[pipes_start])
        self.pipe_end_results = Table2D(PIPE_END_RESULTS, len(ppoints_end), self.time_steps, index = self.ic['pipe']._index_keys[pipes_end])

    def _define_dist_graph_comm(self):
        self.comm = self.global_comm.Create_dist_graph_adjacent(
            sources = self.recv_queue.keys,
            destinations = self.send_queue.keys,
            sourceweights = list(map(len, self.recv_queue.values)),
            destweights = list(map(len, self.send_queue.values)))

    def _define_worker_comm_queues(self):
        points = self.partition['points']['global_idx']
        local_points = self.partition['points']['local_idx']
        pp = self.processors[points]
        pp_idx = np.where(pp != self.rank)[0]
        ppoints = points[pp_idx]

        # Define receive queue
        self.recv_queue = ObjArray()
        for i, p in enumerate(pp_idx):
            if not pp[p] in self.recv_queue.index:
                self.recv_queue[pp[p]] = []
            self.recv_queue[pp[p]].append(ppoints[i])
        # Define send queue
        self.send_queue = ObjArray()
        uboundaries = points[self.where.points['are_uboundaries']]
        dboundaries = points[self.where.points['are_dboundaries']]
        inner = points[self.where.points['are_inner']]

        for p in self.recv_queue.keys:
            self.recv_queue[p] = np.sort(self.recv_queue[p])
            urq = np.isin(self.recv_queue[p], uboundaries)
            drq = np.isin(self.recv_queue[p], dboundaries)
            irq = np.isin(self.recv_queue[p], inner)

            extra_b = np.append(self.recv_queue[p][urq] - 1, self.recv_queue[p][drq] + 1)
            extra_i = np.append(self.recv_queue[p][irq] - 1, self.recv_queue[p][irq] + 1)
            extra = np.append(extra_b, extra_i)
            reduced_extra = extra[np.isin(extra, points)]
            real_extra = [local_points[r] for r in reduced_extra[self.processors[reduced_extra] == self.rank]] # local idx
            if len(real_extra) > 0:
                if not p in self.send_queue.index:
                    self.send_queue[p] = []
                self.send_queue[p].extend(real_extra)
            self.recv_queue[p] = np.sort([local_points[r] for r in self.recv_queue[p]]) # convert to local idx

        for p in self.send_queue.keys:
            self.send_queue[p] = np.sort(self.send_queue[p])

    def _create_selectors(self):
        points = self.partition['points']['global_idx']
        nodes = self.partition['nodes']['global_idx']

        sorter = np.arange(len(points))
        self.where.points['just_in_pipes'] = sorter[np.searchsorted(points, self.partition['nodes']['points'], sorter=sorter)]
        self.where.points['are_tanks'] = np.where(np.isin(points, self.partition['tanks']['points']))[0]
        self.where.points['are_reservoirs'] = np.where(np.isin(points, self.partition['reservoirs']['points']))[0]
        njip = np.cumsum(self.partition['nodes']['context'])
        self.where.nodes['just_in_pipes',] = njip[:-1]
        self.where.nodes['to_points'] = self.where.points['just_in_pipes'][self.where.nodes['just_in_pipes',][:-1]]

        nonpipe = np.isin(self.global_where.points['are_boundaries'], self.global_where.points['are_valve'])
        nonpipe = nonpipe | np.isin(self.global_where.points['are_boundaries'], self.global_where.points['are_pump'])
        local_points = np.isin(self.global_where.points['are_boundaries'], points[self.processors[points] == self.rank])
        dboundary = np.zeros(len(nonpipe), dtype=bool); dboundary[::2] = 1
        uboundary = np.zeros(len(nonpipe), dtype=bool); uboundary[1::2] = 1
        # ---------------------------
        self.where.points['are_uboundaries'] = np.where(np.isin(points, self.global_where.points['are_uboundaries']))[0]
        self.where.points['are_dboundaries'] = np.where(np.isin(points, self.global_where.points['are_dboundaries']))[0]
        self.where.points['are_inner'] = np.setdiff1d(np.arange(len(points), dtype=np.int), \
            np.concatenate((self.where.points['are_uboundaries'], self.where.points['are_dboundaries'])))
        # ---------------------------
        n_pipes = len(self.global_where.points['are_uboundaries'])
        ppipes_idx = np.arange(n_pipes, dtype=int)
        ppipes = np.zeros(n_pipes*2, dtype=int)
        ppipes[::2] = ppipes_idx; ppipes[1::2] = ppipes_idx
        selector_dboundaries = dboundary & (~nonpipe) & local_points
        self.where.points['jip_dboundaries'] = np.where(np.isin(points, self.global_where.points['are_boundaries'][selector_dboundaries]))[0]
        self.where.points['jip_dboundaries',] = ppipes[selector_dboundaries]
        selector_uboundaries = uboundary & (~nonpipe) & local_points
        self.where.points['jip_uboundaries'] = np.where(np.isin(points, self.global_where.points['are_boundaries'][selector_uboundaries]))[0]
        self.where.points['jip_uboundaries',] = ppipes[selector_uboundaries]
        # ---------------------------
        self.where.nodes['just_in_pipes'] = np.arange(len(nodes))
        diff = np.diff(njip)
        self.where.points['just_in_pipes',] = np.array([i for i in range(len(nodes)) for j in range(diff[i])], dtype = int)
        # ---------------------------
        self.where.points['start_inline_valve'] = sorter[np.searchsorted(points, self.partition['inline_valves']['start_points'], sorter=sorter)]
        self.where.points['end_inline_valve'] = sorter[np.searchsorted(points, self.partition['inline_valves']['end_points'], sorter=sorter)]
        self.where.points['start_inline_valve',] = self.partition['inline_valves']['global_idx']
        self.where.points['start_inline_pump'] = sorter[np.searchsorted(points, self.partition['inline_pumps']['start_points'], sorter=sorter)]
        self.where.points['end_inline_pump'] = sorter[np.searchsorted(points, self.partition['inline_pumps']['end_points'], sorter=sorter)]
        self.where.points['start_inline_pump',] = self.partition['inline_pumps']['global_idx']
        self.where.points['are_single_valve'] = sorter[np.searchsorted(points, self.partition['single_valves']['points'], sorter=sorter)]
        self.where.points['are_single_valve',] = self.partition['single_valves']['global_idx']
        self.where.points['are_single_pump'] = sorter[np.searchsorted(points, self.partition['single_pumps']['points'], sorter=sorter)]
        self.where.points['are_single_pump',] = self.partition['single_pumps']['global_idx']

    def define_initial_conditions_for_points(self, points, pipe, start, end):
        q = self.ic['pipe'].flowrate[pipe]
        self.mem_pool_points.flowrate[start:end,0] = q

        start_node = self.ic['pipe'].start_node[pipe]
        start_point = self.global_where.points['are_boundaries'][pipe*2]
        npoints = points - start_point # normalized

        shead = self.ic['node'].head[start_node]

        self.point_properties.B[start:end] = self.ic['pipe'].wave_speed[pipe] / (G * self.ic['pipe'].area[pipe])
        self.point_properties.R[start:end] = self.ic['pipe'].ffactor[pipe] * self.ic['pipe'].dx[pipe] / \
            (2 * G * self.ic['pipe'].diameter[pipe] * self.ic['pipe'].area[pipe] ** 2)
        per_unit_hl = self.ic['pipe'].head_loss[pipe] / self.ic['pipe'].segments[pipe]
        self.mem_pool_points.head[start:end,0] = shead - per_unit_hl*npoints

    def _load_initial_conditions(self):
        points = self.partition['points']['global_idx']
        pipes = self.global_where.points['to_pipes'][points]
        diff = np.where(np.diff(pipes) >= 1)[0] + 1
        if len(diff) > 0:
            for i in range(len(diff)+1):
                if i == 0:
                    start = 0
                    end = diff[i]
                elif i == len(diff):
                    start = diff[i-1]
                    end = None
                else:
                    start = diff[i-1]
                    end = diff[i]
                self.define_initial_conditions_for_points(points[start:end], pipes[start], start, end)
        else:
            self.define_initial_conditions_for_points(points, pipes[0], 0, None)

        self.point_properties.has_plus[self.where.points['are_uboundaries']] = 1
        self.point_properties.has_minus[self.where.points['are_dboundaries']] = 1
        self.point_properties.has_plus[self.where.points['are_inner']] = 1
        self.point_properties.has_minus[self.where.points['are_inner']] = 1

        self.pipe_start_results.inflow[:,0] = self.mem_pool_points.flowrate[self.where.points['are_dboundaries'], 0]
        self.pipe_end_results.outflow[:,0] = self.mem_pool_points.flowrate[self.where.points['are_uboundaries'], 0]
        # self.node_results.head[self.where.nodes['to_points',], 0] = self.mem_pool_points.head[self.where.nodes['to_points'], 0]
        # self.node_results.head[self.where.nodes['to_points',], 0] = self.mem_pool_points.head[self.where.nodes['to_points'], 0]
        # self.node_results.leak_flow[:, 0] = \
        #     self.ic['node'].leak_coefficient * np.sqrt(self.ic['node'].pressure)
        # self.node_results.demand_flow[:, 0] = \
        #     self.ic['node'].demand_coefficient * np.sqrt(self.ic['node'].pressure)

    def exchange_data(self, t):
        t1 = t % 2; t0 = 1 - t1
        send_flow = []
        send_head = []
        for v in self.send_queue.values:
            send_flow.append(self.mem_pool_points.flowrate[v,t0])
            send_head.append(self.mem_pool_points.head[v,t0])
        self._comm_buffer_flow = self.comm.neighbor_alltoall(send_flow)
        self._comm_buffer_head = self.comm.neighbor_alltoall(send_head)
        self.mem_pool_points.flowrate[self._recv_points, t0] = [item for sublist in self._comm_buffer_flow for item in sublist]
        self.mem_pool_points.head[self._recv_points, t0] = [item for sublist in self._comm_buffer_head for item in sublist]

    def run_step(self, t):
        t1 = t % 2; t0 = 1 - t1

        Q0 = self.mem_pool_points.flowrate[:,t0]
        H0 = self.mem_pool_points.head[:,t0]
        Q1 = self.mem_pool_points.flowrate[:,t1]
        H1 = self.mem_pool_points.head[:,t1]

        run_interior_step(
            Q0, H0, Q1, H1,
            self.point_properties.B,
            self.point_properties.R,
            self.point_properties.Cp,
            self.point_properties.Bp,
            self.point_properties.Cm,
            self.point_properties.Bm,
            self.point_properties.has_plus,
            self.point_properties.has_minus)
        run_boundary_step(
            H0, Q1, H1,
            self.node_results.leak_flow[:,t],
            self.node_results.demand_flow[:,t],
            self.point_properties.Cp,
            self.point_properties.Bp,
            self.point_properties.Cm,
            self.point_properties.Bm,
            self.ic['node'].leak_coefficient,
            self.ic['node'].demand_coefficient,
            self.ic['node'].elevation,
            self.where)
        run_valve_step(
            Q1, H1,
            self.point_properties.Cp,
            self.point_properties.Bp,
            self.point_properties.Cm,
            self.point_properties.Bm,
            self.ic['valve'].setting,
            self.ic['valve'].K,
            self.ic['valve'].area,
            self.where)
        run_pump_step(
            self.ic['pump'].source_head,
            Q1, H1,
            self.point_properties.Cp,
            self.point_properties.Bp,
            self.point_properties.Cm,
            self.point_properties.Bm,
            self.ic['pump'].a1,
            self.ic['pump'].a2,
            self.ic['pump'].Hs,
            self.ic['pump'].setting,
            self.where)