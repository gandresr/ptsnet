from multiprocessing.sharedctypes import Value
from re import L
import numpy as np
import os

from tqdm import tqdm
from collections import deque as dq
from pkg_resources import resource_filename
from ptsnet.arrays import ObjArray, Table2D
from ptsnet.simulation.constants import COEFF_TOL, STEP_JOBS, INIT_JOBS, COMM_JOBS
from ptsnet.epanet.util import EN
from ptsnet.utils.data import define_curve, is_array
from ptsnet.utils.io import run_shell, get_root_path
from ptsnet.simulation.init import Initializer
from ptsnet.parallel.comm import CommManager
from ptsnet.parallel.worker import Worker
from ptsnet.results.storage import StorageManager
from ptsnet.results.workspaces import new_workspace_name, list_workspaces, num_workspaces
from ptsnet.simulation.constants import NODE_RESULTS, PIPE_END_RESULTS, PIPE_START_RESULTS, SURGE_PROTECTION_TYPES
from ptsnet.profiler.profiler import Profiler

class PTSNETSettings:
    simplified_settings = (
        'time_step',
        'duration',
        'warnings_on',
        'parallel',
        'gpu',
        'skip_compatibility_check',
        'show_progress',
        'save_results',
        'profiler_on',
        'period',
        'default_wave_speed',
        'wave_speed_file_path',
        'delimiter',
        'wave_speed_method')

    def __init__(self,
        time_step : float = 0.01,
        duration: float = 20,
        warnings_on: bool = False,
        parallel : bool = False,
        gpu : bool = False,
        skip_compatibility_check : bool = False,
        show_progress = False,
        save_results = True,
        profiler_on = False,
        period = 0,
        default_wave_speed = 1000,
        wave_speed_file_path = None,
        delimiter = ',',
        wave_speed_method = 'optimal',
        _super = None):

        self._super = _super
        self.time_step = time_step
        self.duration = duration
        self.time_steps = int(round(duration/time_step))
        self.warnings_on = warnings_on
        self.parallel = parallel
        self.gpu = gpu
        self.skip_compatibility_check = skip_compatibility_check
        self.show_progress = show_progress
        self.save_results = save_results
        self.profiler_on = profiler_on
        self.defined_wave_speeds = False
        self.active_persistance = False
        self.blocked = False
        self.period = period
        self.default_wave_speed = default_wave_speed
        self.wave_speed_file_path = wave_speed_file_path
        self.delimiter = delimiter
        self.wave_speed_method = wave_speed_method
        self.set_default()
        self.settingsOK = True
        self.num_points = None

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
                    if self.warnings_on:
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
                    if self._super != None:
                        lens = sum([len(self._super.element_settings[stype]) for stype in self._super.SETTING_TYPES])
                        if lens > 0:
                            raise ValueError("'%s' can not be modified since settings have been defined" % name)
                    self.time_steps = int(round(self.duration/value))

        object.__setattr__(self, name, value)

    def set_default(self):
        self.is_initialized = False
        self.updated_settings = False

    def to_dict(self, simplified=False):
        l = {}
        for setting, val in self.__dict__.items():
            if setting == '_super': continue
            if simplified and setting not in self.simplified_settings: continue
            l[setting] = val
        return l

class PTSNETCurve:
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
        if is_array(element):
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

class PTSNETSimulation:

    SETTING_TYPES = {
        'valve' : 'valve',
        'pump' : 'pump',
        'burst' : 'node',
        'demand' : 'node',
    }

    def __init__(self, workspace_id = None, inpfile = None, settings = None, init_on = False):
        # Make sure that workspace folder exists
        ws_path = os.path.join(get_root_path(), 'workspaces')
        if not os.path.exists(ws_path):
            os.mkdir(ws_path)
        ### Persistance ----------------------------
        if inpfile == None:
            self.router = CommManager()
            self.settings = PTSNETSettings()
            self.settings.active_persistance = True
            self.results = {}
            self.init_on = init_on
            if workspace_id is None:
                self.workspace_id = num_workspaces() - 1
            else:
                self.workspace_id = workspace_id
            return
        ### ----------------------------------------
        ### New Sim --------------------------------
        if type(settings) != dict:
            print("Warning: using default settings")
            settings = {}
        self.settings = PTSNETSettings(**settings, _super=self)
        self.initializer = Initializer(
            inpfile,
            period = self.settings.period,
            skip_compatibility_check = self.settings.skip_compatibility_check,
            warnings_on = self.settings.warnings_on,
            _super = self)
        self.curves = ObjArray()
        self.element_settings = {type_ : ElementSettings(self) for type_ in self.SETTING_TYPES}
        self.settings.defined_wave_speeds = self.initializer.set_wave_speeds(
            self.settings.default_wave_speed,
            self.settings.wave_speed_file_path,
            self.settings.delimiter,
            self.settings.wave_speed_method
        )
        if self.settings.time_step > self.settings.duration:
            raise ValueError("Duration has to be larger than time step")
        self.initializer.create_selectors()
        self.t = 0
        self.time_stamps = np.array([i*self.settings.time_step for i in range(self.settings.time_steps)])
        self.inpfile = inpfile
        self.settings.num_points = self.initializer.num_points
        # ----------------------------------------
        self.router = CommManager()
        if self.router['main'].size > self.num_points:
            raise ValueError("The number of cores is higher than the number of simulation points")
        self.settings.num_processors = self.router['main'].size
        self.worker = None
        self.results = None
        if self.router['main'].size > 1:
            is_root = self.router['main'].rank == 0
            worspace_name = new_workspace_name(is_root)
            worspace_name = self.router['main'].bcast(worspace_name, root = 0)
            self.storer = StorageManager(worspace_name, router = self.router)
        else:
            self.storer = StorageManager(new_workspace_name())
        # ----------------------------------------
        if (not self.settings.warnings_on) and (self.router['main'].rank == 0) and self.settings.show_progress:
            self.progress = tqdm(total = self.settings.time_steps, position = 0)
            self.progress.update(1)
        # ----------------------------------------
        if self.router['main'].rank == 0:
            self.storer.create_workspace_folders()
        self.router['main'].Barrier()
        self.save_sim_data()

    def __repr__(self):
        return "PTSNETSimulation <duration = %d [s] | time_steps = %d | num_points = %s>" % \
            (self.settings.duration, self.settings.time_steps, format(self.settings.num_points, ',d'))

    def __getitem__(self, index):
        keys = self.results.keys()
        if index == 'time':
            return self.time_stamps
        if not index in keys:
            raise ValueError("not valid label. Use one of the following: %s" % keys)
        return self.results[index]

    def __enter__(self):
        self.load(self.workspace_id)
        self.time_stamps = np.array([i*self.settings.time_step for i in range(self.settings.time_steps)])
        return self

    def __exit__(self, type, value, traceback):
        self.storer.close()

    @property
    def wn(self):
        return self.initializer.wn

    @property
    def ss(self):
        return self.initializer.ss

    @property
    def is_over(self):
        return self.t > self.settings.time_steps - 1

    @property
    def where(self):
        return self.initializer.where

    @property
    def num_points(self):
        return self.initializer.num_points

    @property
    def num_segments(self):
        return self.initializer.num_segments

    @property
    def all_valves(self):
        return self.ss['valve'].labels

    @property
    def all_pumps(self):
        return self.ss['pump'].labels

    @property
    def time_step(self):
        return self.settings.time_step

    def _define_element_setting(self, element, type_, X, Y):
        ic_type = self.SETTING_TYPES[type_]
        if X[0] == 0:
            self.element_settings[type_]._dump_settings(self.ss[ic_type].lloc(element), X[1:], Y[1:])
        else:
            self.element_settings[type_]._dump_settings(self.ss[ic_type].lloc(element), X, Y)

    def run(self):
        while not self.is_over:
            self.run_step()

    def define_valve_operation(self, valve_names, initial_setting, final_setting, start_time=0, end_time=1,
        valve_type='butterfly', function='linear'):

        compatible_valve_types = ('butterfly',)
        compatible_transient_functions = ('linear',)

        if not valve_type in compatible_valve_types:
            raise NotImplementedError("We currently support the following valve types: ", compatible_valve_types)
        if not function in compatible_transient_functions:
            raise NotImplementedError("We currently support the following transient function types: ", compatible_transient_functions)
        if start_time >= end_time:
            raise ValueError("End time must be greater than start time for valve operation")
        if not ((0 <= initial_setting <= 1) and (0 <= final_setting <= 1)):
            raise ValueError("Setting values must be between [0,1]")

        if valve_type == 'butterfly':
            self.add_curve(valve_type, 'valve',
                [1, 0.8, 0.6, 0.4, 0.2, 0],
                [0.067, 0.044, 0.024, 0.011, 0.004, 0.   ])
            self.assign_curve_to(valve_type, valve_names)
            NN = int((end_time-start_time)//self.settings.time_step)
            if function == 'linear':
                vnames = [valve_names] if type(valve_names) == str else valve_names
                if len(vnames) > 0:
                    for valve in vnames:
                        self.define_valve_settings(valve, np.linspace(start_time, end_time, NN), np.linspace(initial_setting, final_setting,NN))

    def define_pump_operation(self, pump_names, initial_setting, final_setting, start_time=0, end_time=1,
        function='linear'):

        compatible_transient_functions = ('linear',)

        if not function in compatible_transient_functions:
            NotImplementedError("We currently support the following transient function types: ", compatible_transient_functions)

        NN = int((end_time-start_time)//self.settings.time_step)
        pnames = [pump_names] if type(pump_names) == str else pump_names

        if function == 'linear':
            if len(pnames) > 0:
                for pump in pnames:
                    self.define_pump_settings(pump, np.linspace(start_time, end_time, NN), np.linspace(initial_setting, final_setting, NN))

    def add_burst(self, node_names, burst_coeff, start_time=0, end_time=1, function='linear'):
        compatible_transient_functions = ('linear',)
        if not function in compatible_transient_functions:
            NotImplementedError("We currently support the following transient function types: ", compatible_transient_functions)

        NN = int((end_time-start_time)//self.time_step)
        nnames = [node_names] if type(node_names) == str else node_names
        if function == 'linear':
            if len(nnames) > 0:
                for node in nnames:
                    self.define_burst_settings(node, np.linspace(start_time, end_time, NN), np.linspace(0, burst_coeff, NN))

    def add_surge_protection(self, node_name, protection_type, tank_area, tank_height=None, water_level=None):
        if not protection_type in ('open', 'closed'):
            raise ValueError(f"Invalid protection type, use ", ('open', 'closed'))
        if node_name in self.ss[f'{protection_type}_protection']:
            raise ValueError(f"The node '{node_name}' is already a surge protection")

        node_id = self.ss['node'].lloc(node_name)
        if (self.ss['node'].degree[node_name] != 2 or
            node_id in self.ss['pump'].start_node or
            node_id in self.ss['pump'].end_node or
            node_id in self.ss['valve'].start_node or
            node_id in self.ss['valve'].end_node):
            raise ValueError(f"The node '{node_name}' is not between 2 pipes")

        if protection_type == 'open':
            self.ss['open_protection'][node_name] = {}
            self.ss['open_protection'][node_name]['node'] = node_id
            self.ss['open_protection'][node_name]['area'] = tank_area
        elif protection_type == 'closed':
            if (tank_height is None) or (water_level is None):
                raise ValueError("Tank height and water level must be defined for closed protection devices")
            self.ss['closed_protection'][node_name] = {}
            self.ss['closed_protection'][node_name]['node'] = node_id
            self.ss['closed_protection'][node_name]['area'] = tank_area
            self.ss['closed_protection'][node_name]['height'] = tank_height
            self.ss['closed_protection'][node_name]['water_level'] = water_level
        else:
            raise ValueError(f"Protection devices can only be of type: {SURGE_PROTECTION_TYPES.keys()}")

    def define_valve_settings(self, valve_name, X, Y):
        self._define_element_setting(valve_name, 'valve', X, Y)

    def define_pump_settings(self, pump_name, X, Y):
        self._define_element_setting(pump_name, 'pump', X, Y)

    def define_burst_settings(self, node_name, X, Y):
        self._define_element_setting(node_name, 'burst', X, Y)

    def define_demand_settings(self, node_name, X, Y):
        self._define_element_setting(node_name, 'demand', X, Y)

    def add_curve(self, curve_name, type_, X, Y):
        if len(self.ss[type_].labels) == 0:
            raise ValueError("There are not elements of type '" + type_ + "' in the model")
        self.curves[curve_name] = PTSNETCurve(X, Y, type_)

    def assign_curve_to(self, curve_name, elements):
        if type(elements) == str:
            elements = [elements]
        if len(elements) == 0:
            raise ValueError("No elements were specified")
        type_ = self.curves[curve_name].type
        for element in elements:
            if self.ss[type_].curve_index[element] == -1:
                if type_ == 'valve':
                    Kv = self.ss[type_].K[element]
                    Kc = self.curves[curve_name](self.ss[type_].setting[element])
                    diff = abs(Kv - Kc)
                    if Kv == 0:
                        self.ss[type_].adjustment[element] = 1
                        self.ss[type_].K[element] = Kc
                    else:
                        self.ss[type_].adjustment[element] = Kv / Kc
                        if diff > COEFF_TOL:
                            if self.settings.warnings_on:
                                print("Warning: the loss coefficient of valve '%s' is not in the curve, the curve will be adjusted" % element)
                N = len(self.curves[curve_name])
                self.ss[type_].curve_index[element] = N
                element_index = self.ss[type_].lloc(element)
                self.curves[curve_name]._add_element(element_index)

    def initialize(self):
        if not self.settings.defined_wave_speeds:
            raise NotImplementedError("wave speed values have not been defined for the pipes")
        for stype in self.SETTING_TYPES:
            self.element_settings[stype]._sort()
        non_assigned_valves = self.ss['valve'].curve_index == -1
        if non_assigned_valves.any():
            self.add_curve('butterfly', 'valve',
                [1, 0.8, 0.6, 0.4, 0.2, 0],
                [0.067, 0.044, 0.024, 0.011, 0.004, 0.   ])
            self.assign_curve_to('butterfly', self.ss['valve'].labels[non_assigned_valves])
            print(f"Warning: valves {self.ss['valve'].labels[non_assigned_valves]} are butterfly by default")
        self.settings.set_default()
        self.t = 0
        self.initializer.create_secondary_elements()
        self.initializer.create_secondary_selectors()
        self._distribute_work()
        self.t = 1
        self.settings.is_initialized = True
        self.save_init_data()

    def _distribute_work(self):
        self.worker = Worker(
            router = self.router,
            num_points = self.num_points,
            where = self.where,
            wn = self.wn,
            ss = self.ss,
            time_step = self.settings.time_step,
            time_steps = self.settings.time_steps,
            inpfile = self.inpfile,
            profiler_on = self.settings.profiler_on)

        self.results = self.worker.results

        # Adding extra communicators
        color_pipe_start = 1 if self.worker.num_start_pipes > 0 else 0
        color_pipe_end = 1 if self.worker.num_end_pipes > 0 else 0
        color_node = 1 if self.worker.num_nodes > 0 else 0
        splitter1 = self.router['main'].Split(color_pipe_start)
        splitter2 = self.router['main'].Split(color_pipe_end)
        splitter3 = self.router['main'].Split(color_node)
        if color_pipe_start:
            self.router.add_communicator('pipe.start', splitter1)
        if color_pipe_end:
            self.router.add_communicator('pipe.end', splitter2)
        if color_node:
            self.router.add_communicator('node', splitter3)

    def run_step(self):
        if not self.settings.is_initialized:
            self.initialize()
        if not self.settings.updated_settings:
            self._update_settings()

        ###
        self.worker.profiler.start('run_step')
        self.worker.run_step(self.t)
        self.worker.profiler.stop('run_step')
        ###

        if self.settings.num_processors > 1:
            ###
            self.worker.profiler.start('barrier1')
            self.router['local'].Barrier()
            self.worker.profiler.stop('barrier1')
            ###

            ###
            self.worker.profiler.start('exchange_data')
            self.worker.exchange_data(self.t)
            self.worker.profiler.stop('exchange_data')
            ###

            ###
            self.worker.profiler.start('barrier2')
            self.router['local'].Barrier()
            self.worker.profiler.stop('barrier2')
            ###

        self.t += 1

        if (not self.settings.warnings_on) and (self.router['main'].rank == 0) and self.settings.show_progress:
            self.progress.update(1)
            if self.is_over:
                self.progress.close()
                print('\n')
        if self.settings.save_results:
            if self.is_over:
                self.save()
        if self.settings.profiler_on:
            if self.is_over:
                self.save_profiler()

    def _set_element_setting(self, type_, element_name, value, step = None, check_warning = False):
        if self.t == 0:
            raise NotImplementedError("simulation has not been initialized")
        if not step is None:
            if not self.can_be_operated(step, check_warning):
                return
        if not is_array(element_name):
            if type(element_name) is str:
                element_name = (element_name,)
                value = [value]
            else:
                raise ValueError("'element_name' should be iterable or str")
        else:
            if not is_array(value):
                value = np.ones(len(element_name)) * value
            elif len(element_name) != len(value):
                raise ValueError("len of 'element_name' array does not match len of 'value' array")

        if type(value) != np.ndarray:
            value = np.array(value)

        if type_ in ('valve', 'pump'):
            if (value > 1).any() or (value < 0).any():
                raise ValueError("setting for %s '%s' not in [0, 1]" % (type_, element_name))
        elif type_ == 'burst':
            if (value < 0).any():
                raise ValueError("burst coefficient has to be >= 0")
        elif type_ == 'demand':
            if (value < 0).any():
                raise ValueError("demand coefficient has to be >= 0")

        ic_type = self.SETTING_TYPES[type_]

        if type(element_name) == np.ndarray:
            if element_name.dtype == np.int:
                if (ic_type == 'node' and type_ == 'burst'):
                    self.ss[ic_type].leak_coefficient[element_name] = value
                else:
                    self.ss[ic_type].setting[element_name] = value
        else:
            if type(element_name) != tuple:
                element_name = tuple(element_name)
            self.ss[ic_type].setting[self.ss[ic_type].lloc(element_name)] = value

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
                self.ss['valve'].K[curve.elements] = \
                    self.ss['valve'].adjustment[curve.elements] * curve(self.ss['valve'].setting[curve.elements])

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

    def save_sim_data(self):
        if self.router['main'].rank == 0:
            self.storer.save_data('inpfile', self.inpfile, comm = 'main')
            self.storer.save_data('initial_conditions', self.ss, comm = 'main')
            self.storer.save_data('settings', self.settings.to_dict(), comm = 'main')

    def save_init_data(self):
        if self.router['main'].rank == 0:
            self.storer.save_data('partitioning', self.worker.partition, comm = 'main')
            self.storer.save_data('local_to_global', self.worker.local_to_global, comm = 'main')

    def save(self):
        if 'pipe.start' in self.router.intra_communicators:
            self.storer.save_data(
                'pipe.start.flowrate',
                self['pipe.start'].flowrate,
                shape = (self.wn.num_pipes, self.settings.time_steps),
                comm = 'pipe.start') # 1
            self.router['pipe.start'].Barrier()

        if 'pipe.end' in self.router.intra_communicators:
            self.storer.save_data(
                'pipe.end.flowrate',
                self['pipe.end'].flowrate,
                shape = (self.wn.num_pipes, self.settings.time_steps),
                comm = 'pipe.end') # 2
            self.router['pipe.end'].Barrier()

        len_nodes = np.sum(self.where.nodes['to_points',] > 0)
        if 'node' in self.router.intra_communicators:
            if self.worker.num_nodes > 0:
                self.storer.save_data(
                    'node.head',
                    self['node'].head,
                    shape = (len_nodes, self.settings.time_steps), # 3
                    comm = 'node')
                self.router['node'].Barrier()

                self.storer.save_data(
                    'node.demand_flow',
                    self['node'].demand_flow,
                    shape = (len_nodes, self.settings.time_steps), # 4
                    comm = 'node')
                self.router['node'].Barrier()

                self.storer.save_data(
                    'node.leak_flow',
                    self['node'].leak_flow,
                    shape = (len_nodes, self.settings.time_steps), # 5
                    comm = 'node')
                self.router['node'].Barrier()

    def save_profiler(self):
        if not self.settings.profiler_on: return

        raw_step_times = np.zeros(
            (len(STEP_JOBS), len(self.worker.profiler.jobs[STEP_JOBS[0]].duration)), dtype=float)
        for i, job in enumerate(STEP_JOBS):
            raw_step_times[i] = self.worker.profiler.jobs[job].duration

        self.storer.save_data(
            'raw_step_times',
            raw_step_times,
            shape = (len(STEP_JOBS)*self.settings.num_processors, raw_step_times.shape[1]),
            comm = 'main')
        self.router['main'].Barrier()

        raw_init_times = np.zeros(
            (len(INIT_JOBS), len(self.worker.profiler.jobs[INIT_JOBS[0]].duration)), dtype=float)
        for i, job in enumerate(INIT_JOBS):
            raw_init_times[i] = self.worker.profiler.jobs[job].duration

        self.storer.save_data(
            'raw_init_times',
            raw_init_times,
            shape = (len(INIT_JOBS)*self.settings.num_processors, raw_init_times.shape[1]),
            comm = 'main')
        self.router['main'].Barrier()

        if self.router['main'].size > 1:
            raw_comm_times = np.zeros(
                (len(COMM_JOBS), len(self.worker.profiler.jobs[COMM_JOBS[0]].duration)), dtype=float)
            for i, job in enumerate(COMM_JOBS):
                raw_comm_times[i] = self.worker.profiler.jobs[job].duration

            self.storer.save_data(
                'raw_comm_times',
                raw_comm_times,
                shape = (len(COMM_JOBS)*self.settings.num_processors, raw_comm_times.shape[1]),
                comm = 'main')
            self.router['main'].Barrier()

    def load(self, workspace_id):
        if self.router['main'].rank == 0:
            wps = list_workspaces()
            if len(wps) < workspace_id:
                raise ValueError(f'The workspace with ID ({workspace_id}) does not exist')

            self.storer = StorageManager(wps[workspace_id], router = self.router)
            settings = self.storer.load_data('settings')
            for i, j in settings.items():
                if not i == 'settingsOK':
                    self.settings.__setattr__(i, j)
                else:
                    continue
            self.profiler = Profiler(_super = self)
            self.profiler.summarize_step_times()
            self.inpfile = self.storer.load_data('inpfile')

            if self.init_on:
                self.initializer = Initializer(
                    self.inpfile,
                    period = self.settings.period,
                    skip_compatibility_check = True,
                    warnings_on = self.settings.warnings_on,
                    _super = self)

            if self.settings.save_results:
                local_to_global = self.storer.load_data('local_to_global')

                node_labels = local_to_global['node']
                sorted_node_labels = sorted(node_labels, key=node_labels.get)
                node_head = self.storer.load_data('node.head')
                node_demand_flow = self.storer.load_data('node.demand_flow')
                node_leak_flow = self.storer.load_data('node.leak_flow')
                self.results['node'] = \
                    Table2D(
                        NODE_RESULTS,
                        node_head.shape[0],
                        node_head.shape[1],
                        labels = sorted_node_labels,
                        allow_replacement = True,
                        persistent = True)
                self.results['node'].head = node_head
                self.results['node'].demand_flow = node_demand_flow
                self.results['node'].leak_flow = node_leak_flow

                pipe_start_labels = local_to_global['pipe.start']
                sorted_pipe_start_labels = sorted(pipe_start_labels, key=pipe_start_labels.get)
                pipe_start_flowrate = self.storer.load_data('pipe.start.flowrate')
                self.results['pipe.start'] = \
                    Table2D(
                        PIPE_START_RESULTS,
                        pipe_start_flowrate.shape[0],
                        pipe_start_flowrate.shape[1],
                        labels = sorted_pipe_start_labels,
                        allow_replacement = True,
                        persistent = True)
                self.results['pipe.start'].flowrate = pipe_start_flowrate

                pipe_end_labels = local_to_global['pipe.end']
                sorted_pipe_end_labels = sorted(pipe_end_labels, key=pipe_end_labels.get)
                pipe_end_flowrate = self.storer.load_data('pipe.end.flowrate')
                self.results['pipe.end'] = \
                    Table2D(
                        PIPE_END_RESULTS,
                        pipe_end_flowrate.shape[0],
                        pipe_end_flowrate.shape[1],
                        labels = sorted_pipe_end_labels,
                        allow_replacement = True,
                        persistent = True)
                self.results['pipe.end'].flowrate = pipe_end_flowrate