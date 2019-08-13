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

NODE_PROPERTIES = {
    'demand' : np.float, #
    'head' : np.float, #
    'pressure' : np.float, #
    'elevation' : np.float, #
    'type' : np.int, #
    'degree' : np.int, #
    'emitter_coefficient' : np.float, #
    'demand_coefficient' : np.float, #
    # ----------------------------------------
    'e_setting_curve_index' : np.int, #
    'd_setting_curve_index' : np.int, #
    'demand_setting' : np.float,
    'emitter_setting' : np.float,
}

PIPE_PROPERTIES = {
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
    'is_inline' : np.bool, #
}

PUMP_PROPERTIES = {
    'start_node' : np.int, #
    'end_node' : np.int, #
    'flowrate' : np.float, #
    'velocity' : np.float, #
    'direction' : np.int, #
    'initial_status' : np.int, #
    'is_inline' : np.bool, #
    # ----------------------------------------
    'A' : np.float, #
    'B' : np.float, #
    'C' : np.float, #
    'curve_index' : np.int,
    'setting_curve_index' : np.int,
    'setting' : np.float,
}

VALVE_PROPERTIES = {
    'start_node' : np.int, #
    'end_node' : np.int, #
    'diameter' : np.float, #
    'area' : np.float, #
    'flowrate' : np.float, #
    'velocity' : np.float, #
    'direction' : np.int, #
    'initial_status' : np.int, #
    'type' : np.int, #
    'is_inline' : np.bool, #
    # ----------------------------------------
    'K' : np.float,
    'setting' : np.float,
    'curve_index' : np.int,
    'setting_curve_index' : np.int,
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

MEM_POOL_POINTS = {
    'flowrate' : np.float,
    'head' : np.float,
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