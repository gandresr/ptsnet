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
    'emitter_coefficient' : np.float, #
    'demand_coefficient' : np.float, #
    'demand' : np.float, #
    'head' : np.float, #
    'pressure' : np.float,
    'elevation' : np.float,
    'type' : np.int, #
}

PIPE_INITIAL_CONDITIONS = {
    'start_node' : np.int, #
    'end_node' : np.int, #
    'length' : np.float, #
    'diameter' : np.float, #
    'area' : np.float, #
    'wave_speed' : np.float, #
    'segments' : np.float, #
    'flowrate' : np.float, #
    'velocity' : np.float, #
    'head_loss' : np.float, #
    'direction' : np.int, #
    'ffactor' : np.float, #
    'dx' : np.float, #
    'type' : np.int, #
}

PUMP_INITIAL_CONDITIONS = {
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
    'start_node' : np.int, #
    'end_node' : np.int, #
    'diameter' : np.float, #
    'area' : np.float, #
    'flowrate' : np.float, #
    'velocity' : np.float, #
    'direction' : np.int, #
    'initial_status' : np.int, #
    'type' : np.int, #
}

MEM_POOL_POINTS = {
    'flowrate' : np.float,
    'head' : np.float,
}

POINT_PROPERTIES = {
    'B' : np.float,
    'R' : np.float,
    'Bm' : np.float,
    'Bp' : np.float,
    'Cm' : np.float,
    'Cp' : np.float,
    'has_plus' : np.int,
    'has_minus' : np.int,
}

NODE_RESULTS = {
    'head' : np.float,
    'emitter_flow' : np.float,
    'demand_flow' : np.float,
}

PIPE_RESULTS = {
    'inflow' : np.float,
    'outflow' : np.float,
}