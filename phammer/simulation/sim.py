import numpy as np
from time import time
from collections import deque as dq
from phammer.simulation.ic import get_initial_conditions, get_water_network
from phammer.arrays.arrays import Table2D, Table, ObjArray
from phammer.simulation.constants import MEM_POOL_POINTS, PIPE_RESULTS, NODE_RESULTS, POINT_PROPERTIES, G, COEFF_TOL
from phammer.arrays.selectors import SelectorSet
from phammer.epanet.util import EN
from phammer.simulation.util import imerge, define_curve, is_iterable
from phammer.simulation.funcs import run_interior_step, run_boundary_step, run_valve_step, run_pump_step
from phammer.simulation.validation import check_compatibility

class HammerSettings:
    def __init__(self,
        time_step : float = 0.01,
        duration: float = 20,
        warnings_on: bool = True,
        parallel : bool = False,
        gpu : bool = False,
        skip_compatibility_check : bool = False,
        _super = None):

        self.settingsOK = False
        self.time_step = time_step
        self.duration = duration
        self.time_steps = int(round(duration/time_step))
        self.warnings_on = warnings_on
        self.parallel = parallel
        self.gpu = gpu
        self.skip_compatibility_check = skip_compatibility_check
        self.defined_wave_speeds = False
        self.set_default()
        self._super = _super
        self.settingsOK = True

    def __repr__(self):
        rep = "\nSimulation settings:\n\n"

        for setting, val in self.__dict__.items():
            if setting == '_super':
                continue
            rep += '%s: %s\n' % (setting, str(val))
        return rep

    def __setattr__(self, name, value):
        try:
            if self.__getattribute__(name) != value:
                if name != 'settingsOK':
                    print("Warning: '%s' value has been changed to %s" % (name, str(value)))
        except:
            pass

        if 'settingsOK' in self.__dict__:
            if self.settingsOK:
                if name == 'duration':
                    self.time_steps = int(round(value/self.time_step))
                elif name == 'time_step':
                    if self.defined_wave_speeds:
                        raise ValueError("'%s' can not be modified since wave speeds have been defined" % name)
                    lens = sum([len(self._super.element_settings[stype]) for stype in self._super.SETTING_TYPES])
                    if lens > 0:
                        raise ValueError("'%s' can not be modified since settings have been defined" % name)
                    self.time_steps = int(round(self.duration/value))

        object.__setattr__(self, name, value)

    def set_default(self):
        self.is_initialized = False
        self.updated_settings = False

class HammerCurve:
    CURVE_TYPES = ('valve', 'pump',)
    def __init__(self, X, Y, type_):
        self.elements = []
        if type_ not in self.CURVE_TYPES:
            raise ValueError("type '%s' is not valid, use ('" % type_ + "', '".join(self.CURVE_TYPES) + "')")
        self.type = type_
        self.X = np.array(X)
        self.Y = np.array(Y)
        order = np.argsort(self.X)
        self.X = self.X[order]
        self.Y = self.Y[order]
        self.fun = define_curve(self.X, self.Y)

    def _add_element(self, element):
        if is_iterable(element):
            for e in element:
                if not element in self.elements:
                    self.elements.append(e)
        else:
            if not element in self.elements:
                self.elements.append(element)

    def __call__(self, value):
        return self.fun(value)

    def __len__(self):
        return len(self.elements)

class ElementSettings:
    ERROR_MSG = "the simulation has started, settings can not be added/modified"
    def __init__(self, _super):
        self.values = []
        self.elements = []
        self.updated = False
        self.activation_times = None
        self.activation_indices = None
        self.is_sorted = False
        self._super = _super

    def __len__(self):
        return len(self.elements)

    def _dump_settings(self, element_index, X, Y):
        X = np.array(X)
        Y = np.array(Y)
        if self.is_sorted:
            raise RuntimeError(self.ERROR_MSG)
        if type(element_index) != int:
            raise ValueError("'element_index' is not int")
        if X.shape != Y.shape:
            raise ValueError("X and Y have different shapes")
        if len(X.shape) > 1:
            raise ValueError("X should be a 1D numpy array")
        xx = X // self._super.settings.time_step
        if len(np.unique(xx)) != len(xx):
            raise ValueError("more than one modification per time step")
        if element_index in self.elements:
            self._remove_entries_of(element_index)

        elist = [element_index for i in range(len(X))]
        self.values += list(zip(elist, xx, Y))
        self.elements.append(element_index)

    def _remove_entries_of(self, element_index):
        if self.is_sorted:
            raise RuntimeError(self.ERROR_MSG)
        self.values = list(filter(lambda x : x[0] != element_index, self.values))
        self.elements.remove(element_index)

    def _sort(self):
        if self.values != [] and not self.is_sorted:
            self.values.sort(key = lambda x : x[1])
            self.values = np.array(self.values, dtype=np.float)
            self.elements = self.values[:,0].astype(np.int)
            self.values = self.values[:,1:]
            self.activation_times, self.activation_indices = np.unique(self.values[:,0].astype(np.int), True)
            self.values = self.values[:,1]
            self.activation_times = dq(self.activation_times)
            self.activation_indices = dq(self.activation_indices)
        self.is_sorted = True

class HammerSimulation:

    SETTING_TYPES = {
        'valve' : 'valve',
        'pump' : 'pump',
        'burst' : 'node',
        'demand' : 'node',
    }

    def __init__(self, inpfile, settings):
        if type(settings) != dict:
            raise TypeError("'settings' are not properly defined, use dict object")
        self.settings = HammerSettings(**settings, _super = self)
        self.wn = get_water_network(inpfile)
        self.ng = self.wn.get_graph()
        self.ic = get_initial_conditions(inpfile, wn = self.wn)
        if not self.settings.skip_compatibility_check:
            if self.settings.warnings_on:
                t = time()
            try:
                check_compatibility(None, wn=self.wn, ic=self.ic)
            except Exception as e:
                print("Elapsed time (model check): ", time() - t, '[s]')
                raise e
            if self.settings.warnings_on:
                print("Success - Compatible Model")
                print("Elapsed time (model check): ", time() - t, '[s]')
        self.curves = ObjArray()
        self.element_settings = {type_ : ElementSettings(self) for type_ in self.SETTING_TYPES}
        self.num_segments = 0
        self.num_points = 0
        self.t = 0

    def __repr__(self):
        return "HammerSimulation <duration = %d [s] | time_steps = %d | num_points = %s>" % \
            (self.settings.duration, self.settings.time_steps, format(self.num_points, ',d'))

    @property
    def is_over(self):
        return self.t > self.settings.time_steps - 1

    def _allocate_memory(self):
        self.mem_pool_points = Table2D(MEM_POOL_POINTS, self.num_points, 2)
        self.point_properties = Table(POINT_PROPERTIES, self.num_points)
        self.pipe_results = Table2D(PIPE_RESULTS, self.wn.num_pipes, self.settings.time_steps, index = self.ic['pipe']._index_keys)
        self.node_results = Table2D(NODE_RESULTS, self.wn.num_nodes, self.settings.time_steps, index = self.ic['node']._index_keys)

    def _create_nonpipe_selectors(self, object_type):
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
        last_order = np.argsort(self.where.points['end_inline_' + object_type,])
        self.where.points['end_inline_' + object_type][:] = self.where.points['end_inline_' + object_type][last_order]
        self.where.points['end_inline_' + object_type,][:] = self.where.points['end_inline_' + object_type,][last_order]
        self.where.points['are_single_' + object_type] = np.sort(self.where.points['are_boundaries'][x3])
        self.where.points['are_single_' + object_type,] = ordered[~self.ic[object_type].is_inline[ordered]]
        last_order = np.argsort(self.where.points['are_single_' + object_type,])
        self.where.points['are_single_' + object_type][:] = self.where.points['are_single_' + object_type][last_order]
        self.where.points['are_single_' + object_type,][:] = self.where.points['are_single_' + object_type,][last_order]

    def _create_selectors(self):
        self.where = SelectorSet(['points', 'pipes', 'nodes', 'valves'])

        # Point, Node and Pipe selectors

        self.where.pipes['to_nodes'] = imerge(self.ic['pipe'].start_node, self.ic['pipe'].end_node)
        self.where.valves['to_nodes'] = imerge(self.ic['valve'].start_node, self.ic['valve'].end_node)
        self.where.nodes['njust_in_pipes'] = np.unique(np.concatenate((
            self.ic['valve'].start_node, self.ic['valve'].end_node,
            self.ic['pump'].start_node, self.ic['pump'].end_node)))
        self.where.points['are_uboundaries'] = np.cumsum(self.ic['pipe'].segments.astype(np.int)+1) - 1
        self.where.points['are_dboundaries'] = self.where.points['are_uboundaries'] - self.ic['pipe'].segments.astype(np.int)
        self.where.points['are_boundaries'] = imerge(self.where.points['are_dboundaries'], self.where.points['are_uboundaries'])
        order = np.argsort(self.where.pipes['to_nodes'])
        nodes, indices = np.unique(self.where.pipes['to_nodes'][order], True)
        self.where.nodes['to_points'] = self.where.points['are_boundaries'][order][indices]
        self.where.nodes['to_points',] = nodes
        self.where.points['are_inner'] = np.setdiff1d(np.arange(self.num_points, dtype=np.int), self.where.points['are_boundaries'])
        x0 = ~np.isin(self.where.pipes['to_nodes'], self.where.nodes['njust_in_pipes'])
        self.where.points['just_in_pipes'] = self.where.points['are_boundaries'][x0]
        self.where.points['just_in_pipes',] = self.where.pipes['to_nodes'][x0]
        order = np.argsort(self.where.points['just_in_pipes',])
        self.where.points['just_in_pipes'] = self.where.points['just_in_pipes'][order]
        self.where.points['just_in_pipes',] = self.where.points['just_in_pipes',][order]
        self.where.points['rjust_in_pipes'] = self.where.points['just_in_pipes']
        self.where.points['rjust_in_pipes',] = np.copy(self.where.points['just_in_pipes',])
        y = self.where.points['rjust_in_pipes',]
        y -= y[0]
        y = (y[1:] - y[:-1]) - 1; y[y < 0] = 0; y = np.cumsum(y)
        self.where.points['rjust_in_pipes',][1:] -= y
        self.where.nodes['just_in_pipes'] = np.unique(self.where.points['just_in_pipes',])
        self.where.nodes['rjust_in_pipes'] = np.unique(self.where.points['rjust_in_pipes',])
        self.where.points['jip_dboundaries'] = self.where.points['are_dboundaries'][x0[0::2]]
        self.where.points['jip_uboundaries'] = self.where.points['are_uboundaries'][x0[1::2]]
        bpoints_types = self.ic['node'].type[self.where.pipes['to_nodes']]
        self.where.points['are_reservoirs'] = self.where.points['are_boundaries'][bpoints_types == EN.RESERVOIR]
        self.where.points['are_tanks'] = self.where.points['are_boundaries'][bpoints_types == EN.TANK]
        bcount = np.bincount(self.where.points['just_in_pipes',])
        bcount = np.cumsum(bcount[bcount != 0]); bcount[1:] = bcount[:-1]; bcount[0] = 0
        self.where.nodes['just_in_pipes',] = bcount

        # Valve selectors
        self._create_nonpipe_selectors('valve')

        # Pump selectors
        self._create_nonpipe_selectors('pump')

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

    def _set_segments(self):
        self.ic['pipe'].segments = self.ic['pipe'].length
        self.ic['pipe'].segments /= self.ic['pipe'].wave_speed

        # Maximum time_step in the system to capture waves in all pipes
        max_dt = min(self.ic['pipe'].segments) / 2 # at least 2 segments in critical pipe

        self.settings.time_step = min(self.settings.time_step, max_dt)

        # The number of segments is defined
        self.ic['pipe'].segments /= self.settings.time_step
        int_segments = np.round(self.ic['pipe'].segments)

        # The wave_speed values are adjusted to compensate the truncation error
        self.ic['pipe'].wave_speed = self.ic['pipe'].wave_speed * self.ic['pipe'].segments/int_segments
        self.ic['pipe'].segments = int_segments
        self.num_segments = int(sum(self.ic['pipe'].segments))
        self.num_points = self.num_segments + self.wn.num_pipes

    def _define_element_setting(self, element, type_, X, Y):
        ic_type = self.SETTING_TYPES[type_]
        self.element_settings[type_]._dump_settings(self.ic[ic_type].iloc(element), X, Y)

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

    def set_wave_speeds(self, default_wave_speed = None, wave_speed_file = None, delimiter=','):
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
            self._set_segments()
            self.settings.defined_wave_speeds = True
            return

        if modified_lines != self.wn.num_pipes:
            self.ic['pipe'].wave_speed[:] = 0
            excep = "The file does not specify wave speed values for all the pipes,\n"
            excep += "it is necessary to define a default wave speed value"
            raise ValueError(excep)

        self._set_segments()
        self.settings.defined_wave_speeds = True

    def define_valve_settings(self, valve_name, X, Y):
        self._define_element_setting(valve_name, 'valve', X, Y)

    def define_pump_settings(self, pump_name, X, Y):
        self._define_element_setting(pump_name, 'pump', X, Y)

    def define_burst_settings(self, node_name, X, Y):
        self._define_element_setting(node_name, 'burst', X, Y)

    def define_demand_settings(self, node_name, X, Y):
        self._define_element_setting(node_name, 'demand', X, Y)

    def set_valve_setting(self, valve_name, value, step = None, check_warning = False):
        self._set_element_setting('valve', valve_name, value, step, check_warning)

    def set_pump_setting(self, pump_name, value, step = None, check_warning = False):
        self._set_element_setting('pump', pump_name, value, step, check_warning)

    def set_burst_setting(self, node_name, value, step = None, check_warning = False):
        self._set_element_setting('burst', node_name, value, step, check_warning)

    def set_demand_setting(self, node_name, value, step = None, check_warning = False):
        self._set_element_setting('demand', node_name, value, step, check_warning)

    def add_curve(self, curve_name, type_, X, Y):
        self.curves[curve_name] = HammerCurve(X, Y, type_)

    def assign_curve_to(self, curve_name, elements):
        if type(elements) == str:
            elements = [elements]
        type_ = self.curves[curve_name].type
        for element in elements:
            if self.ic[type_].curve_index[element] == -1:
                if type_ == 'valve':
                    Kv = self.ic[type_].K[element]
                    Kc = self.curves[curve_name](self.ic[type_].setting[element])
                    diff = abs(Kv - Kc)
                    if Kv == 0:
                        self.ic[type_].adjustment[element] = 1
                        self.ic[type_].K[element] = Kc
                    else:
                        self.ic[type_].adjustment[element] = Kv / Kc
                        if diff > COEFF_TOL:
                            print("Warning: the steady state coefficient of valve '%s' is not in the curve, the curve will be adjusted" % element)
                N = len(self.curves[curve_name])
                self.ic[type_].curve_index[element] = N
                element_index = self.ic[type_].iloc(element)
                self.curves[curve_name]._add_element(element_index)

    def can_be_operated(self, step, check_warning = False):
        check_time = self.t * self.settings.time_step
        if check_time % step >= self.settings.time_step:
            return False
        else:
            if check_warning:
                print("Warning: operating at time time %f" % check_time)
            return True

    def initialize(self):
        if not self.settings.defined_wave_speeds:
            raise NotImplementedError("wave speed values have not been defined for the pipes")
        for stype in self.SETTING_TYPES:
            self.element_settings[stype]._sort()
        non_assigned_valves = self.ic['valve'].curve_index == -1
        if non_assigned_valves.any():
            raise NotImplementedError("it is necessary to assign curves for valves:\n%s" % str(self.ic['valve']._index_keys[non_assigned_valves]))
        self.settings.set_default()
        self.t = 0
        self._create_selectors()
        self._allocate_memory()
        self._load_initial_conditions()
        self.settings.is_initialized = True
        self._update_settings()

    def run_step(self):
        if not self.settings.is_initialized:
            raise NotImplementedError("it is necessary to initialize the simulation before running it")
        if not self.settings.updated_settings:
            self._update_settings()

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
        self.pipe_results.inflow[:,self.t] = self.mem_pool_points.flowrate[self.where.points['are_dboundaries'], t1]
        self.pipe_results.outflow[:,self.t] = self.mem_pool_points.flowrate[self.where.points['are_uboundaries'], t1]
        self.node_results.head[self.where.nodes['to_points',], self.t] = self.mem_pool_points.head[self.where.nodes['to_points'], t1]
        self.t += 1