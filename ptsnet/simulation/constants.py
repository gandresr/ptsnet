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
FILE_TEMPLATE = """
from ptsnet.simulation.sim import PTSNETSimulation
sim = PTSNETSimulation(workspace_id = {workspace_id}, inpfile = '{inpfile}', settings = {settings})\n
"""

# ----- Initial Conditions -----

NODE_PROPERTIES = {
    'demand' : float, #
    'head' : float, #
    'pressure' : float, #
    'elevation' : float, #
    'type' : int, #
    'degree' : int, #
    'processor' : int,
    # ----------------------------------------
    'leak_coefficient' : float, #
    'demand_coefficient' : float, #
}

PIPE_PROPERTIES = {
    'start_node' : int, #
    'end_node' : int, #
    'length' : float, #
    'diameter' : float, #
    'area' : float, #
    'wave_speed' : float, #
    'desired_wave_speed' : float, #
    'segments' : float, #
    'flowrate' : float, #
    'velocity' : float, #
    'head_loss' : float, #
    'direction' : int, #
    'ffactor' : float, #
    'dx' : float, #
    'type' : int, #
    'is_inline' : bool, #
}

PUMP_PROPERTIES = {
    'start_node' : int, #
    'end_node' : int, #
    'flowrate' : float, #
    'velocity' : float, #
    'head_loss' : float, #
    'direction' : int, #
    'initial_status' : int, #
    'is_inline' : bool, #
    'source_head' : float, #
    # ----------------------------------------
    'a1' : float, #
    'a2' : float, #
    'Hs' : float, #
    'curve_index' : int,
    'setting' : float,
}

VALVE_PROPERTIES = {
    'start_node' : int, #
    'end_node' : int, #
    'diameter' : float, #
    'area' : float, #
    'head_loss' : float, #
    'flowrate' : float, #
    'velocity' : float, #
    'direction' : int, #
    'initial_status' : int, #
    'type' : int, #
    'is_inline' : bool, #
    'adjustment' : float, #
    # ----------------------------------------
    'K' : float,
    'setting' : float,
    'curve_index' : int,
}

OPEN_PROTECTION_PROPERTIES = {
    'node' : int, #
    'area' : float, #
    # ----------------------------------------
    'QT' : float, # tank inflow
    'HT0' : float, # tank head t = 0
    'HT1' : float, # tank head t = 1
}

CLOSED_PROTECTION_PROPERTIES = {
    'node' : int, #
    'area' : float, #
    'height' : float,
    'water_level' : float,
    'C' : float, # ideal gas constant
    # ----------------------------------------
    'QT0' : float, # tank inflow t
    'QT1' : float, # tank inflow t + tau
    'HT0' : float, # tank head t
    'HT1' : float, # tank head t + tau
    'HA' : float, # air head
    'VA' : float, # air volume
}

SURGE_PROTECTION_TYPES = {
    'open' : 0,
    'closed' : 1
}

POINT_PROPERTIES = {
    'B' : float,
    'R' : float,
    'Bm' : float,
    'Bp' : float,
    'Cm' : float,
    'Cp' : float,
    'has_plus' : int,
    'has_minus' : int,
}

MEM_POOL_POINTS = {
    'flowrate' : float,
    'head' : float,
}

NODE_RESULTS = {
    'head' : float,
    'leak_flow' : float,
    'demand_flow' : float,
}

PIPE_START_RESULTS = {
    'flowrate' : float,
}

PIPE_END_RESULTS = {
    'flowrate' : float,
}

CLOSED_PROTECTION_RESULTS = {
    'water_level' : float
}

STEP_JOBS = (
    'run_step',
    'run_interior_step',
    'run_general_junction',
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
