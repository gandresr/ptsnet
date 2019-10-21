import numpy as np

from collections import deque
from phammer.simulation.init import Initializator
from phammer.arrays.arrays import Table2D, Table, ObjArray
from phammer.parallel.partitioning import even, get_points
from phammer.simulation.constants import MEM_POOL_POINTS, PIPE_RESULTS, NODE_RESULTS, POINT_PROPERTIES, G, COEFF_TOL
from phammer.simulation.util import is_iterable
from phammer.arrays.selectors import SelectorSet

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
        self.time_steps = kwargs['time_steps']
        self.curves = kwargs['curves']
        self.element_settings = kwargs['element_settings']
        self.t = 0
        self.global_where = kwargs['where']
        self.mem_pool_points = None
        self.point_properties = None
        self.pipe_results = None
        self.node_results = None
        self.where = None
        self.processors = even(self.num_points, self.num_processors)
        self.points = None
        self.receive_data = {}
        self.send_data = {}

        self._create_selectors()
        self._define_worker_points()
        self._allocate_memory()
        self._load_initial_conditions()

    def _allocate_memory(self):
        self.mem_pool_points = Table2D(MEM_POOL_POINTS, len(self.points), 2)
        self.point_properties = Table(POINT_PROPERTIES, len(self.points))
        # self.pipe_results = Table2D(PIPE_RESULTS, self.num_pipes, self.time_steps, index = self.ic['pipe']._index_keys)
        # self.node_results = Table2D(NODE_RESULTS, self.num_nodes, self.time_steps, index = self.ic['node']._index_keys)

    def _define_worker_partition(self):
        self.partition, rcv = get_partition(self.processors, self.rank, self.global_where, self.ic, self.wn)
        rcv_points = self.partition['points'][rcv]
        rcv_processors = self.processors[rcv]
        for k in np.unique(rcv_processors):
            self.receive_data[k] = rcv_points[rcv_processors == k]

    def _create_selectors(self):
        points = self.partition['points']
        nodes = self.partition['nodes']['global_idx']

        sorter = np.argsort(points)
        self.where.points['just_in_pipes'] = sorter[np.searchsorted(points, self.partition['nodes']['points'], sorter=sorter)]
        self.where.points['are_tanks'] = sorter[np.searchsorted(points, self.partition['tanks']['points'], sorter=sorter)]
        self.where.points['are_reservoirs'] = sorter[np.searchsorted(points, self.partition['reservoirs']['points'], sorter=sorter)]
        self.where.nodes['just_in_pipes',] = np.cumsum(self.partition['nodes']['context'])

        nonpipe = np.isin(self.global_where.points['are_boundaries'], self.global_where.points['are_valve'])
        nonpipe = nonpipe | np.isin(self.global_where.points['are_boundaries'], self.global_where.points['are_pump'])
        local_points = np.isin(self.global_where.points['are_boundaries'], points[self.processors[points] == self.rank])
        dboundary = np.zeros(len(nonpipe), dtype=bool); dboundary[::2] = 1
        uboundary = np.zeros(len(nonpipe), dtype=bool); uboundary[1::2] = 1
        # ---------------------------
        self.where.points['jip_dboundaries'] = np.where(np.isin(points, self.global_where.points['are_boundaries'][dboundary & (~nonpipe) & local_points]))[0]
        self.where.points['jip_uboundaries'] = np.where(np.isin(points, self.global_where.points['are_boundaries'][uboundary & (~nonpipe) & local_points]))[0]
        # ---------------------------
        self.where.nodes['just_in_pipes'] = np.arange(len(nodes))
        diff = np.diff(self.where.nodes['just_in_pipes',])
        self.where.points['just_in_pipes',] = np.array([i for i in range(len(nodes)) for j in range(diff[i])])
        # ---------------------------
        self.where.points['start_inline_valve'] = self.partition['inline_valves']['start_points']
        self.where.points['end_inline_valve'] = self.partition['inline_valves']['end_points']
        self.where.points['start_inline_valve',] = self.partition['inline_valves']['global_idx']
        self.where.points['start_inline_pump'] = self.partition['inline_pumps']['start_points']
        self.where.points['end_inline_pump'] = self.partition['inline_pumps']['end_points']
        self.where.points['start_inline_pump',] = self.partition['inline_pumps']['global_idx']
        self.where.points['are_single_valve'] = self.partition['single_valves']['points']
        self.where.points['are_single_valve',] = self.partition['single_valves']['global_idx']
        self.where.points['are_single_pump'] = self.partition['single_pumps']['points']
        self.where.points['are_single_pump',] = self.partition['single_pumps']['global_idx']

    def _load_initial_conditions(self):
        pass

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