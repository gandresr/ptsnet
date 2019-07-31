point_properties = {
    'are_start' : np.bool,
    'are_end' : np.bool,
    'B' : np.float,
    'R' : np.float,
    'Cm' : np.float,
    'Cp' : np.float,
    'Bm' : np.float,
    'Bp' : np.float,
    'flowrate' : np.float,
    'head' : np.float
}

points = Table(point_properties, 5)
print(points.are_start)
points['are_ghost'] = np.array([1,5,6], dtype=np.int)
print(points['are_ghost'])

# ----- Global -----

WARNINGS = True
TIMEIT = True
PARALLEL = False
DEFAULT_FLUID_DENSITY = 997 # kg/m^3
DEFAULT_FFACTOR = 0.035
G = 9.807
TOL = 1E-6

# ----- Index labels for Node tables -----

POINTS_INT = ntuple('POINTS_INT',
    ['subindex',
    'point_type',
    'point_subtype',
    'link_id',
    'processor',
    'is_ghost']
)

POINTS_FLOAT = ntuple('POINTS_FLOAT',
    ['B',
    'R',
    'Cm',
    'Bm',
    'Cp',
    'Bp',
    'has_Cm',
    'has_Cp'])

# ----- Index labels for Node tables -----

NODES_INT = ntuple('NODESunodes
    ['emitter_curve_id',
    'emitter_setting_id']
)

NODES_FLOAT = ntuple('NODES_FLOAT', [
    'demand_coeff',
    'emitter_coeff',
    'emitter_setting'])

# ----- Index labels for Valve tables -----

VALVES_INT = ntuple('VALVES_INT',
    ['upstream_node',
    'downstream_node',
    'curve_id',
    'setting_id']
)

VALVES_FLOAT = ntuple('VALVES_FLOAT',
    ['setting',
    'valve_coeff'])

# ----- Index labels for Pump tables -----

PUMPS_INT = ntuple('PUMPS_INT',
    ['upstream_node',
    'downstream_node',
    'curve_id',
    'setting_id'])

PUMPS_FLOAT = ntuple('PUMPS_FLOAT',
    ['a', 'b', 'c', 'setting', 'max_speed'])

# ----- Types of Elements ----

POINT_TYPES = {
    'interior' : 0,
    'boundary' : 1
}

POINT_SUBTYPES = {
    'none': -1,
    'reservoir': 0,
    'junction': 1,
    'valve': 2,
    'pump': 3
}