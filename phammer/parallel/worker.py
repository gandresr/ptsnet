import numpy as np

from collections import deque
from phammer.simulation.init import Initializator
from phammer.arrays.arrays import Table2D, Table, ObjArray
from phammer.parallel.partitioning import even, get_partition
from phammer.simulation.constants import MEM_POOL_POINTS, PIPE_START_RESULTS, PIPE_END_RESULTS, NODE_RESULTS, POINT_PROPERTIES, G, COEFF_TOL
from phammer.simulation.util import is_iterable
from phammer.arrays.selectors import SelectorSet
from phammer.simulation.funcs import run_boundary_step, run_interior_step, run_pump_step, run_valve_step
class Worker:
    def __init__(self, **kwargs):
        self.send_queue = deque()
        self.recv_queue = deque()
        self.is_initialized = False
        self.comm = kwargs['comm']
        self.rank = kwargs['rank']
        self.num_points = kwargs['num_points']
        self.num_processors = kwargs['num_processors']
        self.wn = kwargs['wn']
        self.ic = kwargs['ic']
        self.settings = kwargs['settings']
        self.time_steps = kwargs['time_steps']
        self.curves = kwargs['curves']
        self.element_settings = kwargs['element_settings']
        self.t = 0
        self.global_where = kwargs['where']
        self.mem_pool_points = None
        self.point_properties = None
        self.pipe_start_results = None
        self.pipe_end_results = None
        self.node_results = None
        self.where = SelectorSet(['points', 'pipes', 'nodes', 'valves', 'pumps'])
        self.processors = even(self.num_points, self.num_processors)
        self.partition = None
        self.receive_data = {}
        self.send_data = {}

        self._define_worker_partition()
        self._create_selectors()
        self._allocate_memory()
        self._load_initial_conditions()

    def _allocate_memory(self):
        self.mem_pool_points = Table2D(MEM_POOL_POINTS, len(self.partition['points']), 2)
        self.point_properties = Table(POINT_PROPERTIES, len(self.partition['points']))

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
        self.node_results = Table2D(NODE_RESULTS, len(nodes), self.time_steps, index = self.ic['node']._index_keys[nodes])

        ppoints_start = self.partition['points'][self.where.points['are_dboundaries']]
        ppoints_end = self.partition['points'][self.where.points['are_uboundaries']]
        pipes_start = self.global_where.points['to_pipes'][ppoints_start]
        pipes_end = self.global_where.points['to_pipes'][ppoints_end]

        self.pipe_start_results = Table2D(PIPE_START_RESULTS, len(ppoints_start), self.time_steps, index = self.ic['pipe']._index_keys[pipes_start])
        self.pipe_end_results = Table2D(PIPE_END_RESULTS, len(ppoints_end), self.time_steps, index = self.ic['pipe']._index_keys[pipes_end])

    def _define_worker_partition(self):
        self.partition, rcv = get_partition(self.processors, self.rank, self.global_where, self.ic, self.wn)
        rcv_points = self.partition['points'][rcv]
        rcv_processors = self.processors[rcv]
        for k in np.unique(rcv_processors):
            self.receive_data[k] = rcv_points[rcv_processors == k]

    def _create_selectors(self):
        points = self.partition['points']
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
        points = self.partition['points']
        pipes = self.global_where.points['to_pipes'][points]
        diff = np.where(np.diff(pipes) >= 1)[0] + 1
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
        self.t = 1

    def run_step(self):
        # if not self.settings.is_initialized:
        #     raise NotImplementedError("it is necessary to initialize the simulation before running it")
        # if not self.settings.updated_settings:
        #     self._update_settings()

        t1 = self.t % 2; t0 = 1 - t1

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
            self.node_results.leak_flow[:,self.t],
            self.node_results.demand_flow[:,self.t],
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
        # self.pipe_results.inflow[:,self.t] = self.mem_pool_points.flowrate[self.where.points['jip_dboundaries'], t1]
        # self.pipe_results.outflow[:,self.t] = self.mem_pool_points.flowrate[self.where.points['jip_uboundaries'], t1]
        # self.node_results.head[self.where.nodes['to_points',], self.t] = self.mem_pool_points.head[self.where.nodes['to_points'], t1]
        self.t += 1

    def _set_element_setting(self, type_, element_name, value, step = None, check_warning = False):
        if self.t == 0:
            raise NotImplementedError("simulation has not been initialized")
        if not step is None:
            if not self.can_be_operated(step, check_warning):
                return
        if not is_iterable(element_name):
            if type(element_name) is str:
                element_name = (element_name,)
                value = [value]
            else:
                raise ValueError("'element_name' should be iterable or str")
        else:
            if not is_iterable(value):
                value = np.ones(len(element_name)) * value
            elif len(element_name) != len(value):
                raise ValueError("len of 'element_name' array does not match len of 'value' array")

        if type(value) != np.ndarray:
            value = np.array(value)

        if type_ in ('valve', 'pump'):
            if (value > 1).any() or (value < 0).any():
                raise ValueError("setting not in [0, 1]" % element)
        elif type_ == 'burst':
            if (value < 0).any():
                raise ValueError("burst coefficient has to be >= 0")
        elif type_ == 'demand':
            if (value < 0).any():
                raise ValueError("demand coefficient has to be >= 0")

        ic_type = self.SETTING_TYPES[type_]

        if type(element_name) == np.ndarray:
            if element_name.dtype == np.int:
                self.ic[ic_type].setting[element_name] = value
        else:
            if type(element_name) != tuple:
                element_name = tuple(element_name)
            self.ic[ic_type].setting[self.ic[ic_type].iloc(element_name)] = value

    def _update_settings(self):
        if not self.settings.updated_settings:
            self._update_coefficients()
            Y = True
            for stype in self.SETTING_TYPES:
                Y &= self.element_settings[stype].updated
            self.settings.updated_settings = Y
            if Y:
                return
            for stype in self.SETTING_TYPES:
                act_times = self.element_settings[stype].activation_times
                act_indices = self.element_settings[stype].activation_indices
                if act_times is None:
                    self.element_settings[stype].updated = True
                    continue
                if len(act_times) == 0:
                    self.element_settings[stype].updated = True
                    continue
                if act_times[0] == 0:
                    act_times.popleft()
                    act_indices.popleft()
                    continue
                if self.t >= act_times[0]:
                    i1 = act_indices[0]
                    i2 = None if len(act_indices) <= 1 else act_indices[1]
                    settings = self.element_settings[stype].values[i1:i2]
                    elements = self.element_settings[stype].elements[i1:i2]
                    self._set_element_setting(stype, elements, settings)
                    act_times.popleft()
                    act_indices.popleft()

    def _update_coefficients(self):
        for curve in self.curves:
            if curve.type == 'valve':
                self.ic['valve'].K[curve.elements] = \
                    self.ic['valve'].adjustment[curve.elements] * curve(self.ic['valve'].setting[curve.elements])

    def set_valve_setting(self, valve_name, value, step = None, check_warning = False):
        self._set_element_setting('valve', valve_name, value, step, check_warning)

    def set_pump_setting(self, pump_name, value, step = None, check_warning = False):
        self._set_element_setting('pump', pump_name, value, step, check_warning)

    def set_burst_setting(self, node_name, value, step = None, check_warning = False):
        self._set_element_setting('burst', node_name, value, step, check_warning)

    def set_demand_setting(self, node_name, value, step = None, check_warning = False):
        self._set_element_setting('demand', node_name, value, step, check_warning)

    def can_be_operated(self, step, check_warning = False):
        check_time = self.t * self.settings.time_step
        if check_time % step >= self.settings.time_step:
            return False
        else:
            if check_warning:
                print("Warning: operating at time time %f" % check_time)
            return True