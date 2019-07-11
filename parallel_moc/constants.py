MOC_PATH = "/home/watsup/Documents/Github/hammer-net/parallel_moc/"

MAX_NEIGHBORS = 6
NULL = -987654321
G = 9.81 # m/s
TOL = 1E-6

# Enums






VALVE_FLOAT = {
    'setting' : 0,
    'area' : 1
}

PUMP_INT = {
    'upstream_junction' : 0,
    'downstream_junction' : 1,
    'curve_id': 2,
    'setting_id': 3
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