MOC_PATH = "/home/watsup/Documents/Github/hammer-net/parallel_moc/"

MAX_NEIGHBORS = 6
NULL = -987654321
G = 9.81 # m/s
TOL = 1E-6

# Enums
NODE_INT = {
    'subindex' : 0,
    'node_type' : 1,
    'link_id' : 2,
    'processor' : 3,
    'is_ghost' : 4,
}

NODE_FLOAT = {
    'B': 0,
    'R': 1,
    'Cm': 2,
    'Cp': 3,
    'Bm': 4,
    'Bp': 5
}

JN = 5

JUNCTION_INT = {
    'downstream_neighbors_num' : 0, # between [0,6]
    'upstream_neighbors_num' : 1, # between [0,6]
    'junction_type' : 2,
    'emitter_curve' : 3,
    'emitter_setting_id': 4,
    'n1' : 5,
    'n2' : 6,
    'n3' : 7,
    'n4' : 8,
    'n5' : 9,
    'n6' : 10
}

JUNCTION_FLOAT = {
    'demand' : 0,
    'head': 1,
    'emitter_coefficient': 2,
    'emitter_setting': 3
}

VALVE_INT = {
    'upstream_junction' : 0,
    'downstream_junction' : 1,
    'curve_id': 2,
    'setting_id' : 3
}

VALVE_FLOAT = {
    'setting' : 0,
    'area' : 1
}

PUMP_INT = {
    'upstream_junction' : 0,
    'downstream_junction' : 1,
    'curve_id': 2,
    'setting_id' : 3
}

PUMP_FLOAT = {
    'a': 0,
    'b': 1,
    'c': 2,
    'setting' : 3,
    'max_speed': 4
}

NODE_TYPES = {
    'none' : NULL,
    'interior' : 0,
    'boundary' : 1
}

JUNCTION_TYPES = {
    'reservoir': 0,
    'pipes': 1,
    'valve': 2,
    'emitter': 3,
    'pump': 4,
    'dead_end': 5
}

CURVE_TYPES = {
    'valve' : 0,
    'pump': 1,
    'emitter' : 2
}
