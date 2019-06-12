
MOC_PATH = "/home/watsup/Documents/Github/hammer-net/parallel_moc/"

NULL = -9876543210
MAX_NEIGHBORS_IN_JUNCTION = 6
G = 9.81 # m/s

# Enums
NODE_INT = {
    'id' : 0,
    'node_type' : 1,
    'link_id' : 2,
    'processor' : 3,
    'is_ghost' : 4,
    'upstream_node' : 5,
    'downstream_node' : 6,
}

NODE_FLOAT = {
    'B': 0,
    'R': 1
}

JUNCTION_INT = {
    'upstream_neighbors_num' : 0, # between [0,6]
    'downstream_neighbors_num' : 1, # between [0,6]
    # Neighbors
    #   first, upstream neighbors are stored and then
    #   downstream neighbors.
    #   * ni: neighbor id
    #   * pi: processor assigned to node with id ni
    'n1' : 2,
    'p1' : 3,
    'n2' : 4,
    'p2' : 5,
    'n3' : 6,
    'p3' : 7,
    'n4' : 8,
    'p4' : 9,
    'n5' : 10,
    'p5' : 11,
    'n6' : 12,
    'p6' : 13
}

LINK_INT = {
    'id': 0,
    'link_type': 1
}

LINK_FLOAT = {
    'diameter' : 0,
    'area' : 1,
    'wave_speed' : 2,
    'ffactor' : 3,
    'length' : 4,
    'dx' : 5,
    'setting': 6
}

NODE_TYPES = {
    'none' : NULL,
    'interior' : 0,
    # boundary nodes
    'reservoir' : 1,
    'junction' : 2,
    'valve_start' : 3,
    'valve_end' : 4,
    'pump_start' : 5,
    'pump_end' : 6
}

LINK_TYPES = {
    'None' : NULL,
    'Pipe': 0,
    'Valve': 1
}