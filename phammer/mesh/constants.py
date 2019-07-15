from collections import namedtuple as ntuple

# ----- Utils -----

WARNINGS = True

# ----- Constants for mesh creation -----

TOL = 1E-6
G = 9.807

# ----- Index labels for Node tables -----

POINTS_INT = ntuple('POINTS_INT',
    ['subindex',
    'point_type',
    'link_id',
    'processor',
    'is_ghost']
)

POINTS_FLOAT = ntuple('POINTS_FLOAT', ['B','R'])

# ----- Index labels for Junction tables -----

NODES_INT = ntuple('NODES_INT',
    ['node_type',
    'emitter_curve_id',
    'emitter_setting_id']
)

NODES_FLOAT = ntuple('NODES_FLOAT',
    ['fixed_demand',
    'demand_coeff',
    'emitter_coeff',
    'emitter_setting']
)

NODES_OBJ = ntuple('NODES_OBJ', ['upstream_points', 'downstream_points'])

# ----- Index labels for Valve tables -----

VALVES_INT = ntuple('VALVES_INT',
    ['upstream_node',
    'downstream_node',
    'curve_id',
    'setting_id']
)

VALVES_FLOAT = ntuple('VALVES_FLOAT', ['setting', 'area'])

# ----- Index labels for Pump tables -----

PUMPS_INT = ntuple('PUMPS_INT',
    ['upstream_node' ,'downstream_node' ,'curve_id','setting_id'])

PUMPS_FLOAT = ntuple('PUMPS_FLOAT',
    ['a', 'b', 'c', 'setting', 'max_speed'])

# ----- Types of Elements ----

POINT_TYPES = {
    'interior' : 0,
    'boundary' : 1
}

NODE_TYPES = {
    'reservoir': 0,
    'junction': 1,
    'valve': 2,
    'pump': 3
}

CURVE_TYPES = {
    'valve': 0,
    'pump': 1,
    'emitter': 2
}