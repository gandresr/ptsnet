import numpy as np
from phammer.simulation.ic import get_initial_conditions, get_water_network
from phammer.arrays.arrays import Table2D

class HammerSettings:
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
        self.settings = HammerSettings(**settings)
        self.wn = get_water_network(inpfile)
        self.ic = get_initial_conditions(inpfile, wn = self.wn)
        self.ng = self.wn.get_graph()
        self.num_segments = 0

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
            return

        if modified_lines != self.wn.num_pipes:
            self.ic['pipes'].wave_speed[:] = 0
            excep = "The file does not specify wave speed values for all the pipes,\n"
            excep += "it is necessary to define a default wave speed value"
            raise ValueError(excep)

    def set_segments(self):
        self.ic['pipes'].segments = self.ic['pipes'].length
        self.ic['pipes'].segments /= self.ic['pipes'].wave_speed

        # Maximum time_step in the system to capture waves in all pipes
        max_dt = min(self.ic['pipes'].segments) / 2 # at least 2 segments in critical pipe

        self.settings.time_step = min(self.settings.time_step, max_dt)

        # The number of segments is defined
        self.ic['pipes'].segments /= self.settings.time_step
        int_segments = np.round(self.ic['pipes'].segments)
        # The wave_speed is adjusted to compensate the truncation error
        self.ic['pipes'].wave_speed = self.ic['pipes'].wave_speed*self.ic['pipes'].segments/int_segments
        self.ic['pipes'].segments = int_segments
        self.num_segments = sum(self.ic['pipes'].segments)