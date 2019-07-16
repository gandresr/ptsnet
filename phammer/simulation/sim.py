import numpy as np

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
    def __init__(self, input_file, duration, time_step, default_wave_speed=None, wave_speed_file = None, delimiter=','):
        if time_step > duration:
            raise Exception("Error: duration < time_step")
        self.time_steps = int(duration/time_step)
        self.t = 0
        self.mesh = Mesh(
            input_file,
            time_step,
            default_wave_speed = default_wave_speed,
            wave_speed_file = wave_speed_file,
            delimiter=delimiter)
        self.steady_state_sim = self.mesh.steady_state_sim
        self.Q = np.zeros((self.time_steps, self.mesh.num_points), dtype=np.float)
        self.H = np.zeros((self.time_steps, self.mesh.num_points), dtype=np.float)
        self.E = np.zeros((self.time_steps, self.mesh.num_points), dtype=np.float)
        self.D = np.zeros((self.time_steps, self.mesh.num_points), dtype=np.float)
        self.Q[0,:], self.H[0,:] = get_initial_conditions(self.mesh)
        self.curves = []
        self.settings = []

