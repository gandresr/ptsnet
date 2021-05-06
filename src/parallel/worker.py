import numpy as np

from collections import defaultdict as ddict
from ptsnet.arrays import Table2D, Table, ObjArray
from ptsnet.parallel.partitioning import even, get_partition
from ptsnet.simulation.constants import MEM_POOL_POINTS, PIPE_START_RESULTS, PIPE_END_RESULTS, NODE_RESULTS, POINT_PROPERTIES, G, COEFF_TOL
from ptsnet.utils.data import is_array
from ptsnet.arrays.selectors import SelectorSet
from ptsnet.simulation.funcs import run_boundary_step, run_interior_step, run_pump_step, run_valve_step
from ptsnet.profiler import Profiler

class Worker:
    def __init__(self, **kwargs):
        self.send_queue = None
        self.recv_queue = None
        self.router = kwargs['router']
        self.wn = kwargs['wn']
        self.ic = kwargs['ic']
        self.global_where = kwargs['where']
        self.time_steps = kwargs['time_steps']
        self.mem_pool_points = None
        self.point_properties = None
        self.num_nodes = 0 # number of nodes in worker
        self.num_start_pipes = 0 # number of start pipes in worker
        self.num_end_pipes = 0 # number of end pipes in worker
        self.num_jip_nodes = 0 # number of just-in-pipes junction nodes in worker
        self.where = SelectorSet(['points', 'pipes', 'nodes', 'valves', 'pumps'])
        self.processors = even(kwargs['num_points'], self.router['main'].size)
        self.is_innactive = False
        innactive_processors = np.empty(self.router['main'].size, dtype=bool)
        self.results = {}
        self.profiler_on = kwargs['profiler_on']
        self.profiler = Profiler(self.router['main'].rank, is_on = self.profiler_on)

        ###
        self.profiler.start('get_partition')
        self.partition = get_partition(
            self.processors, self.router['main'].rank, self.global_where, self.ic,
            self.wn, self.router['main'].size, kwargs['inpfile'])
        self.profiler.stop('get_partition')
        ###

        ###
        self.profiler.start('check_processor_innactivity')
        if self.partition is None:
            self.is_innactive = True
        self.router['main'].Alltoall(np.ones(self.router['main'].size, dtype=bool)*self.is_innactive, innactive_processors)
        if np.any(innactive_processors):
            self.is_innactive = True

            self.profiler.stop('check_processor_innactivity')
            ###
            raise SystemError(" Partitioning is innecficient due to unused processor(s), try executing the parallel routine with less processors")
        self.profiler.stop('check_processor_innactivity')
        ###

        self.points = self.partition['points']['global_idx']
        self.num_points = len(self.points) # ponts assigned to the worker
        self.local_points = np.arange(self.num_points)

        ###
        self.profiler.start('_create_selectors')
        self._create_selectors()
        self.profiler.stop('_create_selectors')
        ###

        ###
        self.profiler.start('_define_worker_comm_queues')
        self._define_worker_comm_queues()
        self.profiler.stop('_define_worker_comm_queues')
        ###

        ###
        self.profiler.start('_define_dist_graph_comm')
        self._define_dist_graph_comm()
        self.profiler.stop('_define_dist_graph_comm')
        ###

        self._comm_buffer_head = []
        self._recv_points = []
        for r in self.recv_queue.values:
            self._comm_buffer_head.append(np.zeros(len(r)))
            self._recv_points.extend(r)
        self._comm_buffer_flow = np.array(self._comm_buffer_head, dtype = object)
        self._comm_buffer_head = np.array(self._comm_buffer_head, dtype = object)

        ###
        self.profiler.start('_allocate_memory')
        if self.router['main'].rank == 0:
            self.local_to_global = {}
        self._allocate_memory()
        self.profiler.stop('_allocate_memory')
        ###

        ###
        self.profiler.start('_load_initial_conditions')
        self._load_initial_conditions()
        self.profiler.stop('_load_initial_conditions')
        ###

    def _allocate_memory(self):
        self.mem_pool_points = Table2D(MEM_POOL_POINTS, self.num_points, 2)
        self.point_properties = Table(POINT_PROPERTIES, self.num_points)
        if self.num_nodes > 0:
            self.results['node'] = Table2D(NODE_RESULTS, self.num_nodes, self.time_steps,
                labels = self.ic['node'].labels[self.where.nodes['all_to_points',]])

        are_my_uboundaries = self.global_where.points['are_uboundaries'] \
            [self.processors[self.global_where.points['are_uboundaries']] == self.router['main'].rank]
        self.where.points['are_my_uboundaries'] = self.local_points[np.isin(self.points, are_my_uboundaries)]

        are_my_dboundaries = self.global_where.points['are_dboundaries'] \
            [self.processors[self.global_where.points['are_dboundaries']] == self.router['main'].rank]
        self.where.points['are_my_dboundaries'] = self.local_points[np.isin(self.points, are_my_dboundaries)]

        ppoints_start = self.points[self.where.points['are_my_dboundaries']]
        ppoints_end = self.points[self.where.points['are_my_uboundaries']]
        pipes_start = self.global_where.points['to_pipes'][ppoints_start]
        pipes_end = self.global_where.points['to_pipes'][ppoints_end]

        self.num_start_pipes = len(ppoints_start)
        self.num_end_pipes = len(ppoints_end)
        if self.num_start_pipes > 0:
            self.results['pipe.start'] = Table2D(PIPE_START_RESULTS, len(ppoints_start), self.time_steps, labels = self.ic['pipe'].labels[pipes_start])
        if self.num_end_pipes > 0:
            self.results['pipe.end'] = Table2D(PIPE_END_RESULTS, len(ppoints_end), self.time_steps, labels = self.ic['pipe'].labels[pipes_end])

        # Root processor gathers indexes to facilitate reading results

        node_indexes = self.router['main'].gather(self.where.nodes['all_to_points',], root = 0)
        pipe_start_indexes = self.router['main'].gather(pipes_start, root = 0)
        pipe_end_indexes = self.router['main'].gather(pipes_end, root = 0)

        if self.router['main'].rank == 0:
            node_indexes = np.concatenate(node_indexes)
            pipe_start_indexes = np.concatenate(pipe_start_indexes)
            pipe_end_indexes = np.concatenate(pipe_end_indexes)

            node_labels = self.ic['node'].labels[node_indexes]
            pipe_start_labels = self.ic['pipe'].labels[pipe_start_indexes]
            pipe_end_labels = self.ic['pipe'].labels[pipe_end_indexes]

            self.local_to_global['node'] = {l : i for i, l in enumerate(node_labels)}
            self.local_to_global['pipe.start'] = {l : i for i, l in enumerate(pipe_start_labels)}
            self.local_to_global['pipe.end'] = {l : i for i, l in enumerate(pipe_end_labels)}

    def _define_dist_graph_comm(self):
        self.router.add_communicator('local', self.router['main'].Create_dist_graph_adjacent(
            sources = self.recv_queue.labels,
            destinations = self.send_queue.labels,
            sourceweights = list(map(len, self.recv_queue.values)),
            destweights = list(map(len, self.send_queue.values))))

    def _define_worker_comm_queues(self):
        local_points = self.partition['points']['local_idx']
        pp = self.processors[self.points]
        pp_idx = np.where(pp != self.router['main'].rank)[0]
        ppoints = self.points[pp_idx]
        # Define receive queue
        self.recv_queue = ObjArray()
        for i, p in enumerate(pp_idx):
            if not pp[p] in self.recv_queue.indexes:
                self.recv_queue[pp[p]] = []
            self.recv_queue[pp[p]].append(ppoints[i])

        # Define send queue
        self.send_queue = ObjArray()
        uboundaries = self.points[self.where.points['are_uboundaries']]
        dboundaries = self.points[self.where.points['are_dboundaries']]
        inner = self.points[self.where.points['are_inner']]

        for p in self.recv_queue.labels:
            self.recv_queue[p] = np.sort(self.recv_queue[p])
            urq = np.isin(self.recv_queue[p], uboundaries)
            drq = np.isin(self.recv_queue[p], dboundaries)
            irq = np.isin(self.recv_queue[p], inner)

            extra_b = np.append(self.recv_queue[p][urq] - 1, self.recv_queue[p][drq] + 1)
            extra_i = np.append(self.recv_queue[p][irq] - 1, self.recv_queue[p][irq] + 1)
            extra = np.append(extra_b, extra_i)
            reduced_extra = extra[np.isin(extra, self.points)]
            real_extra = [local_points[r] for r in reduced_extra[self.processors[reduced_extra] == self.router['main'].rank]] # local idx
            if len(real_extra) > 0:
                if not p in self.send_queue.indexes:
                    self.send_queue[p] = []
                self.send_queue[p].extend(real_extra)
            self.recv_queue[p] = np.sort([local_points[r] for r in self.recv_queue[p]]) # convert to local idx

        for p in self.send_queue.labels:
            self.send_queue[p] = np.sort(np.unique(self.send_queue[p]))

    def _create_selectors(self):
        jip_nodes = self.partition['nodes']['global_idx']
        lpoints = self.partition['points']['local_idx']

        self.where.points['just_in_pipes'] = np.array([lpoints[np] for np in self.partition['nodes']['points']]).astype(int)
        self.where.points['are_tanks'] = np.where(np.isin(self.points, self.partition['tanks']['points']))[0]
        self.where.points['are_reservoirs'] = np.where(np.isin(self.points, self.partition['reservoirs']['points']))[0]
        njip = np.cumsum(self.partition['nodes']['context'])
        self.where.nodes['just_in_pipes',] = njip[:-1]
        self.where.nodes['to_points'] = self.where.points['just_in_pipes'][self.where.nodes['just_in_pipes',][:-1]]

        nonpipe = np.isin(self.global_where.points['are_boundaries'], self.global_where.points['are_valve'])
        nonpipe = nonpipe | np.isin(self.global_where.points['are_boundaries'], self.global_where.points['are_pump'])
        local_points = np.isin(self.global_where.points['are_boundaries'], self.points[self.processors[self.points] == self.router['main'].rank])
        dboundary = np.zeros(len(nonpipe), dtype=bool); dboundary[::2] = 1
        uboundary = np.zeros(len(nonpipe), dtype=bool); uboundary[1::2] = 1
        # ---------------------------
        self.where.points['are_uboundaries'] = np.where(np.isin(self.points, self.global_where.points['are_uboundaries']))[0]
        self.where.points['are_dboundaries'] = np.where(np.isin(self.points, self.global_where.points['are_dboundaries']))[0]
        self.where.points['are_inner'] = np.setdiff1d(np.arange(self.num_points, dtype=np.int), \
            np.concatenate((self.where.points['are_uboundaries'], self.where.points['are_dboundaries'])))
        # ---------------------------
        n_pipes = len(self.global_where.points['are_uboundaries'])
        ppipes_idx = np.arange(n_pipes, dtype=int)
        ppipes = np.zeros(n_pipes*2, dtype=int)
        ppipes[::2] = ppipes_idx; ppipes[1::2] = ppipes_idx
        selector_dboundaries = dboundary & (~nonpipe) & local_points
        self.where.points['jip_dboundaries'] = np.where(np.isin(self.points, self.global_where.points['are_boundaries'][selector_dboundaries]))[0]
        self.where.points['jip_dboundaries',] = ppipes[selector_dboundaries]
        selector_uboundaries = uboundary & (~nonpipe) & local_points
        self.where.points['jip_uboundaries'] = np.where(np.isin(self.points, self.global_where.points['are_boundaries'][selector_uboundaries]))[0]
        self.where.points['jip_uboundaries',] = ppipes[selector_uboundaries]
        # ---------------------------
        diff = np.diff(njip)
        self.where.points['just_in_pipes',] = np.array([i for i in range(len(jip_nodes)) for j in range(diff[i])], dtype = int)
        # ---------------------------
        self.where.points['start_inline_valve'] = np.array([lpoints[spv] for spv in self.partition['inline_valves']['start_points']]).astype(int)
        self.where.points['end_inline_valve'] = np.array([lpoints[epv] for epv in self.partition['inline_valves']['end_points']]).astype(int)
        self.where.points['start_inline_valve',] = self.partition['inline_valves']['global_idx']
        self.where.points['start_inline_pump'] = np.array([lpoints[spp] for spp in self.partition['inline_pumps']['start_points']]).astype(int)
        self.where.points['end_inline_pump'] = np.array([lpoints[epv] for epv in self.partition['inline_pumps']['end_points']]).astype(int)
        self.where.points['start_inline_pump',] = self.partition['inline_pumps']['global_idx']
        self.where.points['are_single_valve'] = np.array([lpoints[svp] for svp in self.partition['single_valves']['points']]).astype(int)
        self.where.points['are_single_valve',] = self.partition['single_valves']['global_idx']
        self.where.points['are_single_pump'] = np.array([lpoints[spp] for spp in self.partition['single_pumps']['points']]).astype(int)
        self.where.points['are_single_pump',] = self.partition['single_pumps']['global_idx']
        # ---------------------------
        nodes = []; node_points = []
        nodes += list(self.partition['nodes']['global_idx'])
        node_points += list(self.partition['nodes']['points'][self.where.nodes['just_in_pipes',]])
        nodes += list(self.partition['tanks']['global_idx'])
        node_points += list(self.partition['tanks']['points'])
        nodes += list(self.partition['reservoirs']['global_idx'])
        node_points += list(self.partition['reservoirs']['points'])
        nodes += list(self.ic['valve'].start_node[self.partition['inline_valves']['global_idx']])
        node_points += list(self.partition['inline_valves']['start_points'])
        nodes += list(self.ic['valve'].end_node[self.partition['inline_valves']['global_idx']])
        node_points += list(self.partition['inline_valves']['end_points'])
        nodes += list(self.ic['pump'].start_node[self.partition['inline_pumps']['global_idx']])
        node_points += list(self.partition['inline_pumps']['start_points'])
        nodes += list(self.ic['pump'].end_node[self.partition['inline_pumps']['global_idx']])
        node_points += list(self.partition['inline_pumps']['end_points'])
        nodes += list(self.ic['valve'].start_node[self.partition['single_valves']['global_idx']])
        node_points += list(self.partition['single_valves']['points'])
        nodes += list(self.ic['pump'].end_node[self.partition['single_pumps']['global_idx']])
        node_points += list(self.partition['single_pumps']['points'])
        nodes = np.array(nodes)
        node_points = np.array(node_points)
        if len(nodes) > 0:
            atp = np.array([lpoints[npoint] for npoint in node_points]).astype(int)
            _, idx_unique = np.unique(nodes, return_index=True)
            sorted_idx = np.sort(idx_unique)
            self.where.nodes['all_to_points'] = atp[sorted_idx]
            self.where.nodes['all_to_points',] = nodes[sorted_idx]
            self.num_nodes = len(self.where.nodes['all_to_points',])
            self.where.nodes['all_just_in_pipes'] = self.partition['nodes']['global_idx']
            self.num_jip_nodes = len(self.where.nodes['all_just_in_pipes'])

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

        if self.num_start_pipes > 0:
            self.results['pipe.start'].flowrate[:,0] = self.mem_pool_points.flowrate[self.where.points['are_my_dboundaries'], 0]
        if self.num_end_pipes > 0:
            self.results['pipe.end'].flowrate[:,0] = self.mem_pool_points.flowrate[self.where.points['are_my_uboundaries'], 0]
        if self.num_nodes > 0:
            self.results['node'].head[:, 0] = self.mem_pool_points.head[self.where.nodes['all_to_points'], 0]
            self.results['node'].leak_flow[:, 0] = \
                self.ic['node'].leak_coefficient[self.where.nodes['all_to_points',]] * \
                    np.sqrt(self.ic['node'].pressure[self.where.nodes['all_to_points',]])
            self.results['node'].demand_flow[:, 0] = \
                self.ic['node'].demand_coefficient[self.where.nodes['all_to_points',]] * \
                    np.sqrt(self.ic['node'].pressure[self.where.nodes['all_to_points',]])

    def exchange_data(self, t):
        t1 = t % 2; t0 = 1 - t1
        send_flow = []
        send_head = []
        for v in self.send_queue.values:
            send_flow.append(self.mem_pool_points.flowrate[v,t1])
            send_head.append(self.mem_pool_points.head[v,t1])
        self._comm_buffer_flow = self.router['local'].neighbor_alltoall(send_flow)
        self._comm_buffer_head = self.router['local'].neighbor_alltoall(send_head)
        self.mem_pool_points.flowrate[self._recv_points, t1] = [item for sublist in self._comm_buffer_flow for item in sublist]
        self.mem_pool_points.head[self._recv_points, t1] = [item for sublist in self._comm_buffer_head for item in sublist]

    def run_step(self, t):
        t1 = t % 2; t0 = 1 - t1

        Q0 = self.mem_pool_points.flowrate[:,t0]
        H0 = self.mem_pool_points.head[:,t0]
        Q1 = self.mem_pool_points.flowrate[:,t1]
        H1 = self.mem_pool_points.head[:,t1]

        ###
        self.profiler.start('run_interior_step')
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
        self.profiler.stop('run_interior_step')
        ###

        ###
        self.profiler.start('run_boundary_step')
        if 'node' in self.results: # worker has junctions
            run_boundary_step(
                H0, Q1, H1,
                self.results['node'].leak_flow[:,t],
                self.results['node'].demand_flow[:,t],
                self.point_properties.Cp,
                self.point_properties.Bp,
                self.point_properties.Cm,
                self.point_properties.Bm,
                self.ic['node'].leak_coefficient,
                self.ic['node'].demand_coefficient,
                self.ic['node'].elevation,
                self.where)
        self.profiler.stop('run_boundary_step')
        ###

        ###
        self.profiler.start('run_valve_step')
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
        self.profiler.stop('run_valve_step')
        ###

        ###
        self.profiler.start('run_pump_step')
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
        self.profiler.stop('run_pump_step')
        ###

        ###
        self.profiler.start('store_results')
        if self.num_start_pipes > 0:
            self.results['pipe.start'].flowrate[:,t] = self.mem_pool_points.flowrate[self.where.points['are_my_dboundaries'], t1]
        if self.num_end_pipes > 0:
            self.results['pipe.end'].flowrate[:,t] = self.mem_pool_points.flowrate[self.where.points['are_my_uboundaries'], t1]
        if self.num_nodes > 0:
            self.results['node'].head[:, t] = self.mem_pool_points.head[self.where.nodes['all_to_points'], t1]
        self.profiler.stop('store_results')
        ###


