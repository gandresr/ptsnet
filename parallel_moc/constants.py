
MOC_PATH = "/home/watsup/Documents/Github/hammer-net/parallel_moc/"

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

JUNCTION_INT = {
    'downstream_neighbors_num' : 0, # between [0,6]
    'upstream_neighbors_num' : 1, # between [0,6]
    'junction_type' : 2
}

def define_junctions_int_table(degree):
    for i in range(degree):
        # Neighbors
        #   first, downstream neighbors are stored and then
        #   upstream neighbors.
        #   * ni: neighbor id
        #   * pi: processor assigned to node with id ni
        JUNCTION_INT['n%d' % (i+1)] = 3 + i

JUNCTION_FLOAT = {
    'demand' : 0,
    'head': 1
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
    'pump': 3,
    'dead_end': 4,
    'burst': 5,
    'leakage': 6,
}

CURVE_TYPES = {
    'pump': 0,
    'valve' : 1
}