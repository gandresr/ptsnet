import numpy as np

from phammer.simulation.ic import get_initial_conditions, get_water_network
from phammer.arrays.arrays import Table2D, Table
from phammer.simulation.constants import MEM_POOL_POINTS, PIPE_RESULTS, NODE_RESULTS, POINT_PROPERTIES, G
from phammer.arrays.selectors import SelectorSet
from phammer.epanet.util import EN
from phammer.simulation.util import imerge
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
                if name == 'settingsOK':
                    return
                print("Warning: '%s' value has been changed to %s" % (name, str(value)))
        except:
            pass
        object.__setattr__(self, name, value)
        if (name == 'duration' or name == 'time_step') and self.settingsOK:
            self.time_steps = int(self.duration/self.time_step)

class HammerSimulation:
    def __init__(self, inpfile, settings):
        if type(settings) != dict:
            raise TypeError("'settings' are not properly defined, use dict object")
        self.settings = HammerSettings(**settings, _super = self)
        self.wn = get_water_network(inpfile)
        self.ic = get_initial_conditions(inpfile, wn = self.wn)
        self.ng = self.wn.get_graph()
        self.num_segments = 0
        self.num_points = 0
        self.t = 0

    def __repr__(self):
        return "HammerSimulation <duration = %d [s] | time_steps = %d | num_points = %s>" % \
            (self.settings.duration, self.settings.time_steps, format(self.num_points, ',d'))

    def _allocate_memory(self):
        self.mem_pool_points = Table2D(MEM_POOL_POINTS, self.num_points, 2)
        self.point_properties = Table(POINT_PROPERTIES, self.num_points)
        self.pipe_results = Table2D(PIPE_RESULTS, self.wn.num_pipes, self.settings.time_steps)
        self.node_results = Table2D(NODE_RESULTS, self.wn.num_nodes, self.settings.time_steps)

    def _create_selectors(self):
        self.where = SelectorSet(['points', 'pipes'])
        self.where.points['are_uboundaries'] = np.cumsum(self.ic['pipes'].segments.astype(np.int)+1) - 1
        self.where.points['are_dboundaries'] = self.where.points['are_uboundaries'] - self.ic['pipes'].segments.astype(np.int)
        self.where.pipes['to_nodes'] = imerge(self.ic['pipes'].start_node, self.ic['pipes'].end_node)
        bpoints_types = self.ic['nodes'].type[self.where.pipes['to_nodes']]
        self.where.points['are_boundaries'] = imerge(self.where.points['are_dboundaries'], self.where.points['are_uboundaries'])
        self.where.points['are_inner'] = np.setdiff1d(np.arange(self.num_points, dtype=np.int), self.where.points['are_boundaries'])
        self.where.points['are_reservoirs'] = self.where.points['are_boundaries'][bpoints_types == EN.RESERVOIR]
        self.where.points['are_tanks'] = self.where.points['are_boundaries'][bpoints_types == EN.TANK]
        self.where.points['are_junctions'] = self.where.points['are_boundaries'][bpoints_types == EN.JUNCTION]
        self.where.points['to_junctions'] = self.where.pipes['to_nodes'][bpoints_types == EN.JUNCTION]
        order = np.argsort(self.where.points['to_junctions'])
        self.where.points['are_junctions'] = self.where.points['are_junctions'][order]
        self.where.points['to_junctions'] = self.where.points['to_junctions'][order]
        bindices = np.cumsum(np.bincount(self.where.points['to_junctions']))
        self.where.points['are_junctions',] = np.zeros(len(bindices))
        self.where.points['are_junctions',][1:] = bindices[:-1]

    def _load_initial_conditions(self):
        self.mem_pool_points.head[self.where.points['are_boundaries'], 0] = self.ic['nodes'].head[self.where.pipes['to_nodes']]
        per_unit_hl = self.ic['pipes'].head_loss / self.ic['pipes'].segments
        for i in range(self.wn.num_pipes):
            k = self.where.points['are_dboundaries'][i]
            s = int(self.ic['pipes'].segments[i])
            self.mem_pool_points.head[k:k+s+1, 0] = self.mem_pool_points.head[k,0] - (per_unit_hl[i] * np.arange(s+1))
            self.mem_pool_points.flowrate[k:k+s+1, 0] = self.ic['pipes'].flowrate[i]

    def set_segments(self):
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
        # self._update_moc_constants()

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
            self.set_segments()
            return

        if modified_lines != self.wn.num_pipes:
            self.ic['pipes'].wave_speed[:] = 0
            excep = "The file does not specify wave speed values for all the pipes,\n"
            excep += "it is necessary to define a default wave speed value"
            raise ValueError(excep)

        self.settings.defined_wave_speeds = True
        self.set_segments()

    def initialize(self):
        self._create_selectors()
        self._allocate_memory()
        self._load_initial_conditions()