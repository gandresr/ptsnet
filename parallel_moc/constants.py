
MOC_PATH = "/home/watsup/Documents/Github/hammer-net/parallel_moc/"

NULL = -10
MAX_NEIGHBORS_IN_JUNCTION = 6
G = 9.81 # m/s

# Enums
NODE_INT = {
    'id' : 0,
    'subindex': 1,
    'node_type' : 2,
    'link_id' : 3,
    'processor' : 4,
    'is_ghost' : 5,
}

NODE_FLOAT = {
    'B': 0,
    'R': 1
}

JUNCTION_INT = {
    'downstream_neighbors_num' : 0, # between [0,6]
    'upstream_neighbors_num' : 1, # between [0,6]
    # Neighbors
    #   first, downstream neighbors are stored and then
    #   upstream neighbors.
    #   * ni: neighbor id
    #   * pi: processor assigned to node with id ni
    'n1' : 2,
    'n2' : 3,
    'n3' : 4,
    'n4' : 5,
    'n5' : 6,
    'n6' : 7,
    'p1' : 8,
    'p2' : 9,
    'p3' : 10,
    'p4' : 11,
    'p5' : 12,
    'p6' : 13
}

JUNCTION_FLOAT = {
    'demand' : 0,
    'head': 1
}

LINK_INT = {
    'id': 0,
    'link_type': 1,
    'setting': 2,
    'curve': 3,
    'curve_type': 4
}

LINK_FLOAT = {
    'diameter' : 0,
    'area' : 1,
    'wave_speed' : 2,
    'ffactor' : 3,
    'length' : 4,
    'dx' : 5
}

NODE_TYPES = {
    'none' : NULL,
    'interior' : 0,
    # boundary nodes
    'junction' : 1,
    'valve' : 2,
    'pump' : 3
}

LINK_TYPES = {
    'None' : NULL,
    'Pipe': 0,
    'Valve': 1,
    'Pump': 2
}

CURVE_TYPES = {
    'Valve': 0 # Cd vs % open
}