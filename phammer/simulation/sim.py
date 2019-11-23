import numpy as np

from time import time
from collections import deque as dq
from phammer.arrays.arrays import Table2D, Table, ObjArray
from phammer.simulation.constants import COEFF_TOL
from phammer.epanet.util import EN
from phammer.simulation.util import define_curve, is_iterable, run_shell
from phammer.simulation.init import Initializator
from pkg_resources import resource_filename
from phammer.parallel.worker import Worker
from mpi4py import MPI

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

    def __init__(self, inpfile, settings, default_wave_speed = None, wave_speed_file = None, delimiter=','):
        if type(settings) != dict:
            raise TypeError("'settings' are not properly defined, use dict object")
        self.settings = HammerSettings(**settings, _super=self)
        self.initializator = Initializator(
            inpfile,
            self.settings.skip_compatibility_check,
            self.settings.warnings_on,
            _super = self)
        self.curves = ObjArray()
        self.element_settings = {type_ : ElementSettings(self) for type_ in self.SETTING_TYPES}
        self.settings.defined_wave_speeds = self.initializator.set_wave_speeds(default_wave_speed, wave_speed_file, delimiter)
        self.initializator.create_selectors()
        self.t = 0
        # ----------------------------------------
        self.comm = MPI.COMM_WORLD
        if self.comm.size > self.num_points:
            raise ValueError("The number of cores is higher than the number of simulation points")
        self.settings.num_processors = self.comm.size
        self.rank = self.comm.Get_rank()
        self.worker = None

    def __repr__(self):
        return "HammerSimulation <duration = %d [s] | time_steps = %d | num_points = %s>" % \
            (self.settings.duration, self.settings.time_steps, format(self.initializator.num_points, ',d'))

    @property
    def wn(self):
        return self.initializator.wn

    @property
    def ic(self):
        return self.initializator.ic

    @property
    def is_over(self):
        return self.t > self.settings.time_steps - 1

    @property
    def where(self):
        return self.initializator.where

    @property
    def num_points(self):
        return self.initializator.num_points

    @property
    def num_segments(self):
        return self.initializator.num_segments

    def _define_element_setting(self, element, type_, X, Y):
        ic_type = self.SETTING_TYPES[type_]
        self.element_settings[type_]._dump_settings(self.ic[ic_type].iloc(element), X, Y)

    def define_valve_settings(self, valve_name, X, Y):
        self._define_element_setting(valve_name, 'valve', X, Y)

    def define_pump_settings(self, pump_name, X, Y):
        self._define_element_setting(pump_name, 'pump', X, Y)

    def define_burst_settings(self, node_name, X, Y):
        self._define_element_setting(node_name, 'burst', X, Y)

    def define_demand_settings(self, node_name, X, Y):
        self._define_element_setting(node_name, 'demand', X, Y)

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
                            if self.settings.warnings_on:
                                print("Warning: the steady state coefficient of valve '%s' is not in the curve, the curve will be adjusted" % element)
                N = len(self.curves[curve_name])
                self.ic[type_].curve_index[element] = N
                element_index = self.ic[type_].iloc(element)
                self.curves[curve_name]._add_element(element_index)

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
        self._distribute_work()
        self.t = 1
        self.settings.is_initialized = True

    def _distribute_work(self):
        self.worker = Worker(
            rank = self.rank,
            comm = self.comm,
            num_points = self.num_points,
            num_processors = self.comm.size,
            where = self.where,
            wn = self.wn,
            ic = self.ic,
            time_steps = self.settings.time_steps)

    def run_step(self):
        if not self.settings.is_initialized:
            raise NotImplementedError("it is necessary to initialize the simulation before running it")
        if not self.settings.updated_settings:
            self._update_settings()
        self.worker.run_step(self.t)
        if self.settings.num_processors > 1:
            self.comm.Barrier()
            self.worker.exchange_data(self.t)
            self.worker.comm.Barrier()
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