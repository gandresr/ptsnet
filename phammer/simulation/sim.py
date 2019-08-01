import numpy as np
from phammer.simulation.ic import get_initial_conditions
from phammer.arrays.table import Table2D
from phammer.simulation.constants import RESULTS

class Settings:
    def __init__(self,
        time_step : float = 0.01,
        duration: float = 20,
        warnings_on: bool = True,
        parallel : bool = False,
        gpu : bool = False):

        self.time_step = time_step
        self.duration = duration
        self.time_steps = int(duration/time_step)
        self.warnings_on = warnings_on
        self.parallel = parallel
        self.gpu = gpu

    def __repr__(self):
        rep = "\nSimulation settings:\n\n"

        for setting, val in self.__dict__.items():
            rep += '%s: %s\n' % (setting, str(val))
        return rep

    def __setattr__(self, name, value):
        try:
            if self.__getattribute__(name) != value:
                print("Warning: '%s' value has been changed to %s" % (name, str(value)))
        except:
            pass
        object.__setattr__(self, name, value)

class HammerSimulation:
    def __init__(self, inpfile, settings):
        if type(settings) != dict:
            raise TypeError("'settings' are not properly defined, use dict object")
        self.settings = Settings(**settings)
        self.ic = get_initial_conditions(inpfile)
        self.num_points = 0
        self.results = Table(RESULTS, self.num_points)

    def set_wave_speeds(self, default_wave_speed = None, wave_speed_file = None, delimiter=','):
        # wave_speeds = np.

        if default_wave_speed is None and wave_speed_file is None:
            raise Exception("Wave speed values not specified")

        if default_wave_speed is not None:
            wave_speeds = dict.fromkeys(self.ss.pipes, default_wave_speed)

        if wave_speed_file:
            with open(wave_speed_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    pipe, wave_speed = line.split(delimiter)
                    wave_speeds[pipe] = float(wave_speed)

        if len(wave_speeds) != self.ss.num_pipes:
            wave_speeds = {}
            raise Exception("""
            The file does not specify wave speed values for all the pipes,
            it is necessary to define a default wave speed value""")

        return wave_speeds