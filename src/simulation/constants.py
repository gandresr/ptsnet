import numpy as np

# ----- Global -----

WARNINGS = True
TIMEIT = True
PARALLEL = False
CEIL_FFACTOR = 10
FLOOR_FFACTOR = 1e-6
DEFAULT_FFACTOR = 0.035
G = 9.807 # SI gravity
TOL = 1E-6
COEFF_TOL = 1E-6

# ----- Initial Conditions -----

NODE_PROPERTIES = {
    'demand' : np.float, #
    'head' : np.float, #
    'pressure' : np.float, #
    'elevation' : np.float, #
    'type' : np.int, #
    'degree' : np.int, #
    'processor' : np.int,
    # ----------------------------------------
    'leak_coefficient' : np.float, #
    'demand_coefficient' : np.float, #
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
    'head_loss' : np.float, #
    'direction' : np.int, #
    'initial_status' : np.int, #
    'is_inline' : np.bool, #
    'source_head' : np.float, #
    # ----------------------------------------
    'a1' : np.float, #
    'a2' : np.float, #
    'Hs' : np.float, #
    'curve_index' : np.int,
    'setting' : np.float,
}

VALVE_PROPERTIES = {
    'start_node' : np.int, #
    'end_node' : np.int, #
    'diameter' : np.float, #
    'area' : np.float, #
    'head_loss' : np.float, #
    'flowrate' : np.float, #
    'velocity' : np.float, #
    'direction' : np.int, #
    'initial_status' : np.int, #
    'type' : np.int, #
    'is_inline' : np.bool, #
    'adjustment' : np.float, #
    # ----------------------------------------
    'K' : np.float,
    'setting' : np.float,
    'curve_index' : np.int,
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
    'leak_flow' : np.float,
    'demand_flow' : np.float,
}

PIPE_START_RESULTS = {
    'flowrate' : np.float,
}

PIPE_END_RESULTS = {
    'flowrate' : np.float,
}

STEP_JOBS = (
    'run_step',
    'run_interior_step',
    'run_boundary_step',
    'run_valve_step',
    'run_pump_step',
    'store_results',
)

INIT_JOBS = (
    'get_partition',
    '_create_selectors',
    '_define_dist_graph_comm',
    '_allocate_memory',
    '_load_initial_conditions'
)

COMM_JOBS = (
    'exchange_data',
    'barrier1',
    'barrier2'
)