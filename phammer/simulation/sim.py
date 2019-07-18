import numpy as np
import wntr

from phammer.simulation.utils import define_curve, is_iterable
from phammer.mesh.mesh import Mesh
from phammer.simulation.initial_conditions import get_initial_conditions
from phammer.simulation.utils import set_settings, set_coefficients

class Simulation:
    """
    Here all the tables and properties required to
    run a MOC simulation are defined. Tables for
    simulations in parallel are created

    In the meantime:
    * valves are not valid between two junctions
    * it is not possible to connect one valve to another
    """
    def __init__(self, input_file, duration, time_step, default_wave_speed = None, wave_speed_file = None, delimiter=',', full_results=False):
        if time_step > duration:
            raise Exception("Error: duration < time_step")

        self.t = 0 # current time
        self.time_step = time_step
        self.time_steps = int(duration/time_step)
        self.sim_range = range(1, len(self.time_steps))
        self.full_results = full_results

        self.fname = input_file[:input_file.find('.inp')]
        self.wn = wntr.network.WaterNetworkModel(input_file)
        self.steady_state_sim = None

        # Mesh is initialized here - updated later when running first step
        self.mesh = Mesh(
            self.fname + '.inp',
            self.time_step,
            self.wn,
            default_wave_speed = default_wave_speed,
            wave_speed_file = wave_speed_file,
            delimiter = delimiter)

        if full_results:
            self.Q = np.zeros((self.time_steps, self.mesh.num_points), dtype=np.float)
            self.H = np.zeros((self.time_steps, self.mesh.num_points), dtype=np.float)
        else:
            self.Q0 = np.zeros(self.mesh.num_points, dtype=np.float)
            self.H0 = np.zeros(self.mesh.num_points, dtype=np.float)
            self.Q = np.zeros((self.time_steps, 2*self.mesh.wn.num_pipes), dtype=np.float)
            self.H = np.zeros((self.time_steps, 2*self.mesh.wn.num_pipes), dtype=np.float)

        self.E = np.zeros((self.time_steps, self.mesh.num_points), dtype=np.float)
        self.D = np.zeros((self.time_steps, self.mesh.num_points), dtype=np.float)

        # store pairs (obj_id, scipy_interpolator)
        self.pump_curves = {}
        self.valve_curves = {}
        self.emitter_curves = {}

        # store pairs (obj_id, settings array)
        self.pump_settings = {}
        self.valve_settings = {}
        self.emitter_settings = {}

    def add_emitter(self, node, area, discharge_coeff, initial_setting=1):
        if not (0 <= initial_setting <= 1):
            raise Exception("Initial setting should be in [0, 1]")
        emitter_node = self.wn.get_node(node)
        emitter_node.add_leak(self.wn, area=area, discharge_coeff=discharge_coeff*initial_setting, start_time=0)
        self.mesh.properties['int']['nodes'].emmitter_setting[self.mesh.node_ids[node]] = initial_setting

    def set_pump_setting(self, pump, setting):
        self.mesh.properties['int']['pumps'].setting[self.mesh.pump_ids[pump]] = setting

    def set_valve_setting(self, valve, setting):
        self.mesh.properties['int']['valves'].setting[self.mesh.valve_ids[valve]] = setting

    def set_emitter_setting(self, node, setting):
        self.mesh.properties['int']['nodes'].emitter_setting[self.mesh.node_ids[node]] = setting

    def _define_settings(self, obj_id, obj_type, obj_settings):
        if is_iterable(obj_settings):
            if not all(0 <= x <= 1 for x in obj_settings):
                raise Exception("Setting values should be in [0, 1]")
            if obj_type == 'pumps':
                self.mesh.properties['int'][obj_type].setting_id[obj_id] = len(self.pump_settings)
                self.pump_settings[obj_id] = obj_settings
            elif obj_type == 'valves':
                self.mesh.properties['int'][obj_type].setting_id[obj_id] = len(self.valve_settings)
                self.valve_settings[obj_id] = obj_settings
            elif obj_type == 'nodes':
                self.mesh.properties['int'][obj_type].setting_id[obj_id] = len(self.emitter_settings)
                self.emitter_settings[obj_id] = obj_settings
            else:
                raise Exception("Type error: obj_type not compatible (internal error)")
        else:
            raise Exception("Type error: setting type should be iterable")

    def define_pump_settings(self, pump, settings):
        self._define_settings(self.mesh.pump_ids[pump], 'pumps', settings)

    def define_valve_settings(self, valve, settings):
        self._define_settings(self.mesh.valve_ids[valve], 'valves', settings)

    def define_emitter_settings(self, node, settings):
        self._define_settings(self.mesh.node_ids[node], 'nodes', settings)

    def define_pump_curve(self, pump, X, Y):
        pump_id = self.mesh.pump_ids[pump]
        self.mesh.properties['int']['pumps'].pump_curve_id[pump_id] = len(self.pump_curves)
        self.pump_curves[pump_id] = define_curve(X, Y)

    def define_valve_curve(self, valve, X, Y):
        valve_id = self.mesh.valve_ids[valve]
        self.mesh.properties['int']['valves'].curve_id[valve_id] = len(self.valve_curves)
        self.valve_curves.[valve_id] = define_curve(X, Y)

    def define_emitter_curve(self, node, X, Y):
        node_id = self.mesh.node_ids[node]
        self.mesh.properties['int']['nodes'].emitter_curve_id[node_id] = len(self.emitter_curves)
        self.emitter_curves[node_id] = define_curve(X, Y)

    def start(self):
        self.mesh.create_mesh()
        self.steady_state_sim = self.mesh.steady_state_sim
        if self.full_results:
            self.Q[0,:], self.H[0,:] = get_initial_conditions(self.mesh)
        else:
            self.Q0, self.H0 = get_initial_conditions(self.mesh)
        self.t += 1

    def _update_settings(self):
        set_settings(self.t, self.pump_settings, self.mesh.properties['float']['pumps'].setting)
        set_settings(self.t, self.valve_settings, self.mesh.properties['float']['valves'].setting)
        set_coefficients(
            self.valve_curves,
            self.mesh.properties['float']['valves'].valve_coeff,
            self.mesh.properties['float']['valves'].setting)
        set_settings(self.t, self.emitter_settings, self.mesh.properties['float']['nodes'].setting)
        set_coefficients(
            self.emitter_curves,
            self.mesh.properties['float']['nodes'].emitter_coeff,
            self.mesh.properties['float']['nodes'].setting)