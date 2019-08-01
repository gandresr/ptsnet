import numpy as np

# ----- Global -----

WARNINGS = True
TIMEIT = True
PARALLEL = False
DEFAULT_FLUID_DENSITY = 997 # kg/m^3
CEIL_FFACTOR = 0.035
FLOOR_FFACTOR = 1E-4
DEFAULT_FFACTOR = 0.035
G = 9.807 # SI gravity
TOL = 1E-6

# ----- Initial Conditions -----

NODE_INITIAL_CONDITIONS = {
    'ID' : '<U50', #
    'emitter_coefficient' : np.float, #
    'demand' : np.float, #
    'head' : np.float, #
}

PIPE_INITIAL_CONDITIONS = {
    'ID' : '<U50', #
    'start_node' : np.int, #
    'end_node' : np.int, #
    'length' : np.float, #
    'diameter' : np.float, #
    'area' : np.float, #
    'wave_speed' : np.float,
    'flowrate' : np.float, #
    'velocity' : np.float, #
    'head_loss' : np.float, #
    'direction' : np.int, #
    'ffactor' : np.float, #
    'B' : np.float,
    'R' : np.float,
}

PUMP_INITIAL_CONDITIONS = {
    'ID' : '<U50', #
    'start_node' : np.int, #
    'end_node' : np.int, #
    'flowrate' : np.float, #
    'velocity' : np.float, #
    'direction' : np.int, #
    'initial_status' : np.int, #
    'A' : np.float, #
    'B' : np.float, #
    'C' : np.float, #
}

VALVE_INITIAL_CONDITIONS = {
    'ID' : '<U50', #
    'start_node' : np.int, #
    'end_node' : np.int, #
    'diameter' : np.float, #
    'area' : np.float, #
    'flowrate' : np.float, #
    'velocity' : np.float, #
    'direction' : np.int, #
    'initial_status' : np.int, #
}