import numpy as np

# ----- Global -----

WARNINGS = True
TIMEIT = True
PARALLEL = False
DEFAULT_FLUID_DENSITY = 997 # kg/m^3
DEFAULT_FFACTOR = 0.035
G = 9.807 # SI gravity
TOL = 1E-6

# ----- Initial Conditions -----

NODE_INITIAL_CONDITIONS = {
    'ID' : '<U3',
    'emitter_coefficient' : np.float,
    'demand' : np.float,
    'head' : np.float,
}

LINK_INITIAL_CONDITIONS = {
    'ID' : '<U3',
    'start_node' : np.int,
    'end_node' : np.int,
    'length' : np.float,
    'diameter' : np.float,
    'area' : np.float,
    'wave_speed' : np.float,
    'flowrate' : np.float,
    'direction' : np.int,
    'ffactor' : np.float,
    'B' : np.float,
    'R' : np.float,
}