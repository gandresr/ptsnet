
MOC_PATH = "/home/watsup/Documents/Github/hammer-net/parallel_moc/"

NULL = -987654321
MAX_NEIGHBORS_IN_JUNCTION = NULL
G = 9.81 # m/s

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
    'R': 1
}

JUNCTION_INT = {
    'downstream_neighbors_num' : 0, # between [0,6]
    'upstream_neighbors_num' : 1, # between [0,6]
    'junction_type' : 2
}

def define_junctions_int_table(degree):
    _N_JUNCTION_INT = len(JUNCTION_INT)
    for i in range(degree):
        # Neighbors
        #   first, downstream neighbors are stored and then
        #   upstream neighbors.
        #   * ni: neighbor id
        #   * pi: processor assigned to node with id ni
        JUNCTION_INT['n%d' % (i+1)] = _N_JUNCTION_INT + i

JUNCTION_FLOAT = {
    'demand' : 0,
    'head': 1
}

VALVE_INT = {
    'upstream_junction' : 0,
    'downstream_junction' : 1,
    'setting_id' : 2,
    'curve_id': 3
}

VALVE_FLOAT = {

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
    'max_speed': 0,
}

NODE_TYPES = {
    'none' : NULL,
    'interior' : 0,
    # boundary nodes
    'junction' : 1,
    'valve' : 2,
    'pump' : 3
}

JUNCTION_TYPES = {
    'pipe': 0,
    'valve': 1,
    'pump': 2,
    'dead_end': 3,
    'burst': 4,
    'leakage': 5,
}