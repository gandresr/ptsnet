import numpy as np

from phammer.simulation.ic import get_initial_conditions, get_water_network
from phammer.arrays.arrays import Table2D, Table, ObjArray
from phammer.simulation.constants import MEM_POOL_POINTS, PIPE_RESULTS, NODE_RESULTS, POINT_PROPERTIES, G
from phammer.arrays.selectors import SelectorSet
from phammer.epanet.util import EN
from phammer.simulation.util import imerge, define_curve, is_iterable
from phammer.simulation.funcs import run_interior_step, run_boundary_step

class HammerSettings:
    def __init__(self,
        time_step : float = 0.01,
        duration: float = 20,
        warnings_on: bool = True,
        parallel : bool = False,
        gpu : bool = False,
        _super = None):

        self.settingsOK = False
        self.time_step = time_step
        self.duration = duration
        self.time_steps = int(duration/time_step)
        self.warnings_on = warnings_on
        self.parallel = parallel
        self.gpu = gpu
        self.defined_wave_speeds = False
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
                    self.time_steps = int(value/self.time_step)
                elif name == 'time_step':
                    self.time_steps = int(self.duration/value)

        object.__setattr__(self, name, value)

class HammerCurve:
    types = [
        'valve_curve',
        'valve_setting',
        'pump_curve',
        'pump_setting',
        'emitter_curve',
        'emitter_setting',
        'demand',
        'demand_setting']

    def __init__(self, X, Y, type_):
        self.elements = []
        if type_ not in self.types:
            raise ValueError("type is not valid, use ('" + "', '".join(self.types) + "')")
        self.type = type_
        self.X = np.array(X)
        self.Y = np.array(Y)
        order = np.argsort(self.X)
        self.X = self.X[order]
        self.Y = self.Y[order]
        if self.type in ('valve_curve', 'pump_curve', 'emitter_curve'):
            self.fun = define_curve(self.X, self.Y)
        else:
            self.fun = None

    def add_element(self, element):
        if not element in self.elements:
            self.elements.append(element)

    def __iadd__(self, element):
        self.add_element(element)

class HammerSimulation:
    def __init__(self, inpfile, settings):
        if type(settings) != dict:
            raise TypeError("'settings' are not properly defined, use dict object")
        self.settings = HammerSettings(**settings, _super = self)
        self.wn = get_water_network(inpfile)
        self.ic = get_initial_conditions(inpfile, wn = self.wn)
        self.ng = self.wn.get_graph()
        self.curves = ObjArray()
        self.num_segments = 0
        self.num_points = 0
        self.t = 0

    def __repr__(self):
        return "HammerSimulation <duration = %d [s] | time_steps = %d | num_points = %s>" % \
            (self.settings.duration, self.settings.time_steps, format(self.num_points, ',d'))

    def _allocate_memory(self):
        self.mem_pool_points = Table2D(MEM_POOL_POINTS, self.num_points, 2)
        self.point_properties = Table(POINT_PROPERTIES, self.num_points)
        self.pipe_results = Table2D(PIPE_RESULTS, self.wn.num_pipes, self.settings.time_steps, index = self.ic['pipes']._index_keys)
        self.node_results = Table2D(NODE_RESULTS, self.wn.num_nodes, self.settings.time_steps, index = self.ic['nodes']._index_keys)

    def _create_selectors(self):
        self.where = SelectorSet(['points', 'pipes', 'nodes'])

        # Point, None and pipe selectors
        self.where.pipes['to_nodes'] = imerge(self.ic['pipes'].start_node, self.ic['pipes'].end_node)
        self.where.nodes['njust_in_pipes'] = np.unique(np.concatenate((
            self.ic['valves'].start_node, self.ic['valves'].end_node,
            self.ic['pumps'].start_node, self.ic['pumps'].end_node)))
        self.where.points['are_uboundaries'] = np.cumsum(self.ic['pipes'].segments.astype(np.int)+1) - 1
        self.where.points['are_dboundaries'] = self.where.points['are_uboundaries'] - self.ic['pipes'].segments.astype(np.int)
        self.where.points['are_boundaries'] = imerge(self.where.points['are_dboundaries'], self.where.points['are_uboundaries'])
        self.where.points['are_inner'] = np.setdiff1d(np.arange(self.num_points, dtype=np.int), self.where.points['are_boundaries'])
        x = ~np.isin(self.where.pipes['to_nodes'], self.where.nodes['njust_in_pipes'])
        self.where.points['just_in_pipes'] = self.where.points['are_boundaries'][x]
        self.where.points['just_in_pipes',] = self.where.pipes['to_nodes'][x]
        order = np.argsort(self.where.points['just_in_pipes',])
        self.where.points['just_in_pipes'] = self.where.points['just_in_pipes'][order]
        self.where.points['just_in_pipes',] = self.where.points['just_in_pipes',][order]
        self.where.points['rjust_in_pipes'] = self.where.points['just_in_pipes']
        self.where.points['rjust_in_pipes',] = np.copy(self.where.points['just_in_pipes',])
        y = self.where.points['rjust_in_pipes',]
        y = (y[1:] - y[:-1]) - 1; y[y < 0] = 0; y = np.cumsum(y)
        self.where.points['rjust_in_pipes',][1:] -= y
        self.where.nodes['just_in_pipes'] = np.unique(self.where.points['just_in_pipes',])
        self.where.nodes['rjust_in_pipes'] = np.unique(self.where.points['rjust_in_pipes',])
        self.where.points['jip_dboundaries'] = self.where.points['are_dboundaries'][x[0::2]]
        self.where.points['jip_uboundaries'] = self.where.points['are_uboundaries'][x[1::2]]
        bpoints_types = self.ic['nodes'].type[self.where.pipes['to_nodes']]
        self.where.points['are_reservoirs'] = self.where.points['are_boundaries'][bpoints_types == EN.RESERVOIR]
        self.where.points['are_tanks'] = self.where.points['are_boundaries'][bpoints_types == EN.TANK]
        bcount = np.bincount(self.where.points['just_in_pipes',])
        bcount = np.cumsum(bcount[bcount != 0]); bcount[1:] = bcount[:-1]; bcount[0] = 0
        self.where.nodes['just_in_pipes',] = bcount
        x = np.isin(self.where.pipes['to_nodes'], self.ic['valves'].start_node[self.ic['valves'].is_inline])
        self.where.points['in_valves'] = self.where.points['are_boundaries'][x]
        x = np.isin(self.where.pipes['to_nodes'], self.ic['valves'].end_node[self.ic['valves'].is_inline])
        self.where.points['out_valves'] = self.where.points['are_boundaries'][x]
        x = np.isin(self.where.pipes['to_nodes'], self.ic['pumps'].start_node[self.ic['pumps'].is_inline])
        self.where.points['in_pumps'] = self.where.points['are_boundaries'][x]
        x = np.isin(self.where.pipes['to_nodes'], self.ic['pumps'].end_node[self.ic['pumps'].is_inline])
        self.where.points['out_pumps'] = self.where.points['are_boundaries'][x]

    def _load_initial_conditions(self):
        self.mem_pool_points.head[self.where.points['are_boundaries'], 0] = self.ic['nodes'].head[self.where.pipes['to_nodes']]
        self.ic['pipes'].dx = self.ic['pipes'].length / self.ic['pipes'].segments
        per_unit_hl = self.ic['pipes'].head_loss / self.ic['pipes'].segments
        self.point_properties.has_plus[self.where.points['are_uboundaries']] = 1
        self.point_properties.has_plus[self.where.points['are_inner']] = 1
        self.point_properties.has_minus[self.where.points['are_dboundaries']] = 1
        self.point_properties.has_minus[self.where.points['are_inner']] = 1
        for i in range(self.wn.num_pipes):
            k = self.where.points['are_dboundaries'][i]
            s = int(self.ic['pipes'].segments[i])
            self.mem_pool_points.head[k:k+s+1, 0] = self.mem_pool_points.head[k,0] - (per_unit_hl[i] * np.arange(s+1))
            self.mem_pool_points.flowrate[k:k+s+1, 0] = self.ic['pipes'].flowrate[i]
            self.point_properties.B[k:k+s+1] = self.ic['pipes'].wave_speed[i] / (G * self.ic['pipes'].area[i])
            self.point_properties.R[k:k+s+1] = self.ic['pipes'].ffactor[i] * self.ic['pipes'].dx[i] / \
                (2 * G * self.ic['pipes'].diameter[i] * self.ic['pipes'].area[i] ** 2)

    def _set_segments(self):
        self.ic['pipes'].segments = self.ic['pipes'].length
        self.ic['pipes'].segments /= self.ic['pipes'].wave_speed

        # Maximum time_step in the system to capture waves in all pipes
        max_dt = min(self.ic['pipes'].segments) / 2 # at least 2 segments in critical pipe

        self.settings.time_step = min(self.settings.time_step, max_dt)

        # The number of segments is defined
        self.ic['pipes'].segments /= self.settings.time_step
        int_segments = np.round(self.ic['pipes'].segments)

        # The wave_speed values are adjusted to compensate the truncation error
        self.ic['pipes'].wave_speed = self.ic['pipes'].wave_speed * self.ic['pipes'].segments/int_segments
        self.ic['pipes'].segments = int_segments
        self.num_segments = int(sum(self.ic['pipes'].segments))
        self.num_points = self.num_segments + self.wn.num_pipes

    def set_wave_speeds(self, default_wave_speed = None, wave_speed_file = None, delimiter=','):
        if default_wave_speed is None and wave_speed_file is None:
            raise ValueError("wave_speed was not specified")

        if not default_wave_speed is None:
            self.ic['pipes'].wave_speed[:] = default_wave_speed

        modified_lines = 0
        if not wave_speed_file is None:
            with open(wave_speed_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if len(line) <= 1:
                        raise ValueError("The wave_speed file has to have to entries per line 'pipe,wave_speed'")
                    pipe, wave_speed = line.split(delimiter)
                    self.ic['pipes'].wave_speed[pipe] = float(wave_speed)
                    modified_lines += 1
        else:
            self.settings.defined_wave_speeds = True
            self._set_segments()
            return

        if modified_lines != self.wn.num_pipes:
            self.ic['pipes'].wave_speed[:] = 0
            excep = "The file does not specify wave speed values for all the pipes,\n"
            excep += "it is necessary to define a default wave speed value"
            raise ValueError(excep)

        self.settings.defined_wave_speeds = True
        self._set_segments()

    def add_curve(self, curve_name, type_, X, Y):
        self.curves[curve_name] = HammerCurve(X, Y, type_)

    def assign_curve_to(self, curve_name, elements):
        if type(elements) == str:
            elements = [elements]
        for element in elements:
            if element in self.wn.valve_name_list:
                if self.curves[curve_name].type != 'valve_curve':
                    raise ValueError("only curves of type 'valve_curve' can be defined for valve %s" % element)
            elif (element in self.wn.pump_name_list):
                if self.curves[curve_name].type != 'pump_curve':
                    raise ValueError("only curves of type 'pump_curve' can be defined for pump %s" % element)
            elif element in self.wn.node_name_list:
                if not self.curves[curve_name].type in ('emitter_curve', 'emitter_setting', 'demand', 'demand_setting'):
                    raise ValueError("only emitter and demand curves can be defined for node %s" % element)
            else:
                raise ValueError("element is not valid")
            self.curves[curve_name].add_element(element)

    def initialize(self):
        self._create_selectors()
        self._allocate_memory()
        self._load_initial_conditions()

    def run_step(self):
        t0 = self.t % 2
        t1 = 1 - t0
        run_interior_step(
            self.mem_pool_points.flowrate[:,t0],
            self.mem_pool_points.head[:,t0],
            self.mem_pool_points.flowrate[:,t1],
            self.mem_pool_points.head[:,t1],
            self.point_properties.B,
            self.point_properties.R,
            self.point_properties.Cp,
            self.point_properties.Bp,
            self.point_properties.Cm,
            self.point_properties.Bm,
            self.point_properties.has_plus,
            self.point_properties.has_minus)
        run_boundary_step(
            self.mem_pool_points.head[:,t0],
            self.mem_pool_points.flowrate[:,t1],
            self.mem_pool_points.head[:,t1],
            self.node_results.emitter_flow[:,self.t],
            self.node_results.demand_flow[:,self.t],
            self.point_properties.Cp,
            self.point_properties.Bp,
            self.point_properties.Cm,
            self.point_properties.Bm,
            self.ic['nodes'].emitter_coefficient,
            self.ic['nodes'].demand_coefficient,
            self.ic['nodes'].elevation,
            self.where)

        self.t += 1
        self.pipe_results.inflow[:,self.t] = self.mem_pool_points.flowrate[self.where.points['are_dboundaries'],t1]
        self.pipe_results.outflow[:,self.t] = self.mem_pool_points.flowrate[self.where.points['are_uboundaries'],t1]