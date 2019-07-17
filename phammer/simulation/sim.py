import numpy as np

from phammer.simulation.utils import define_curve
from phammer.mesh.mesh import Mesh
from phammer.simulation.constants import *
from phammer.simulation.initial_conditions import get_initial_conditions

class Simulation:
    """
    Here all the tables and properties required to
    run a MOC simulation are defined. Tables for
    simulations in parallel are created

    In the meantime:
    * valves are not valid between two junctions
    * it is not possible to connect one valve to another
    """
    def __init__(self, input_file, duration, time_step, full_results=False):
        if time_step > duration:
            raise Exception("Error: duration < time_step")
        self.time_steps = int(duration/time_step)
        self.t = 0
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
        self.mesh = None
        self.curves = []
        self.settings = []

    def define_valve_curve(self, valve, X, Y):
        valve_id = self.mesh.valve_ids[valve]
        self.mesh.properties['int']['valves'].curve_id[valve_id] = len(self.curves)
        self.curves.append(define_curve(X, Y))

    def define_emitter_curve(self, node, X, Y):
        node_id = self.mesh.node_ids[node]
        self.mesh.properties['int']['nodes'].emitter_curve_id[node_id] = len(self.curves)
        self.curves.append(define_curve(X, Y))

    def define_valve_setting(self, valve, setting):
        if setting < 0 or setting > 1:
            raise Exception("Setting value has to be in [0,1]")
        valve_id = self.mesh.valve_ids[valve]
        self.mesh.properties['float']['valves'].setting[valve_id] = setting

    def define_emitter_setting(self, node, setting):
        if setting < 0 or setting > 1:
            raise Exception("Setting value has to be in [0,1]")
        node_id = self.mesh.node_ids[node]
        self.mesh.properties['float']['nodes'].emitter_setting[node_id] = setting

    def define_valve(self,):
        pass

    def define_pump(self, ):
        pass

    def initialize(self, default_wave_speed = None, wave_speed_file = None, delimiter=',')
        self.mesh = Mesh(
            input_file,
            time_step,
            default_wave_speed = default_wave_speed,
            wave_speed_file = wave_speed_file,
            delimiter = delimiter)
        self.steady_state_sim = self.mesh.steady_state_sim
        self.Q[0,:], self.H[0,:] = get_initial_conditions(self.mesh)