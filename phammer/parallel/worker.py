from collections import deque
from phammer.simulation.init import Initializator
from phammer.arrays.arrays import Table2D, Table, ObjArray
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
        self.wn = kwargs['wn']
        self.start_index = kwargs['start_index']
        self.ic = kwargs['ic']
        self.time_steps = kwargs['time_steps']
        self.curves = kwargs['curves']
        self.element_settings = kwargs['element_settings']
        self.t = 0
        self.mem_pool_points = None
        self.point_properties = None
        self.pipe_results = None
        self.node_results = None
        self.where = None

        self._allocate_memory()
        self._create_selectors(kwargs['where'])
        # self._load_initial_conditions()
        print(self.rank)
        print(self.where.points)

    def _allocate_memory(self):
        self.mem_pool_points = Table2D(MEM_POOL_POINTS, self.num_points, 2)
        self.point_properties = Table(POINT_PROPERTIES, self.num_points)
        self.pipe_results = Table2D(PIPE_RESULTS, self.wn.num_pipes, self.time_steps, index = self.ic['pipe']._index_keys)
        self.node_results = Table2D(NODE_RESULTS, self.wn.num_nodes, self.time_steps, index = self.ic['node']._index_keys)

    def _create_selectors(self, where):
        self.where = SelectorSet(where.categories)
        for category in where.categories:
            sset = where.__dict__[category].__dict__
            selectors = list(sset['_selectors'].keys())
            for selector in selectors:
                s = sset['_selectors'][selector]
                try:
                    c = sset['_contexts'][selector] # context
                except:
                    c = None
                components = (s >= self.start_index) & (s < self.start_index + self.num_points)
                self.where.__dict__[category].__dict__['_selectors'][selector] = s[components]
                if not c is None:
                    self.where.__dict__[category].__dict__['_contexts'][selector] = c[components]

    def _load_initial_conditions(self):
        self.mem_pool_points.head[self.where.points['are_boundaries'], 0] = self.ic['node'].head[self.where.pipes['to_nodes']]
        self.ic['pipe'].dx = self.ic['pipe'].length / self.ic['pipe'].segments
        per_unit_hl = self.ic['pipe'].head_loss / self.ic['pipe'].segments
        self.point_properties.has_plus[self.where.points['are_uboundaries']] = 1
        self.point_properties.has_plus[self.where.points['are_inner']] = 1
        self.point_properties.has_minus[self.where.points['are_dboundaries']] = 1
        self.point_properties.has_minus[self.where.points['are_inner']] = 1
        for i in range(self.wn.num_pipes):
            k = self.where.points['are_dboundaries'][i]
            s = int(self.ic['pipe'].segments[i])
            self.mem_pool_points.head[k:k+s+1, 0] = self.mem_pool_points.head[k,0] - (per_unit_hl[i] * np.arange(s+1))
            xx = np.argwhere(self.mem_pool_points.head[k:k+s+1, 0] < 0).T
            self.mem_pool_points.flowrate[k:k+s+1, 0] = self.ic['pipe'].flowrate[i]
            self.point_properties.B[k:k+s+1] = self.ic['pipe'].wave_speed[i] / (G * self.ic['pipe'].area[i])
            self.point_properties.R[k:k+s+1] = self.ic['pipe'].ffactor[i] * self.ic['pipe'].dx[i] / \
                (2 * G * self.ic['pipe'].diameter[i] * self.ic['pipe'].area[i] ** 2)
        self.pipe_results.inflow[:,0] = self.mem_pool_points.flowrate[self.where.points['are_dboundaries'], 0]
        self.pipe_results.outflow[:,0] = self.mem_pool_points.flowrate[self.where.points['are_uboundaries'], 0]
        self.node_results.head[self.where.nodes['to_points',], 0] = self.mem_pool_points.head[self.where.nodes['to_points'], 0]
        self.node_results.head[self.where.nodes['to_points',], 0] = self.mem_pool_points.head[self.where.nodes['to_points'], 0]
        self.node_results.leak_flow[:, 0] = \
            self.ic['node'].leak_coefficient * np.sqrt(self.ic['node'].pressure)
        self.node_results.demand_flow[:, 0] = \
            self.ic['node'].demand_coefficient * np.sqrt(self.ic['node'].pressure)
        self.t = 1

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