
MOC_PATH = "/home/watsup/Documents/Github/hammer-net/parallel_moc/"

NULL = -9876543210
# Enums
NODE = {
    'id' : 0,
    'node_type' : 1, # {none, reservoir, junction, end, valve_a, valve_b}
    'link_id' : 2, # {none (-1), link_id, valve_id}
    'processor' : 3,
    'is_ghost' : 4,
    'upstream_neighbors_num' : 5, # between [0,6]
    'downstream_neighbors_num' : 6, # between [0,6]
    # Neighbors
    #   first, upstream neighbors are stored and then
    #   downstream neighbors.
    #   * ni: neighbor id
    #   * pi: processor assigned to node with id ni
    'n1' : 7,
    'p1' : 8,
    'n2' : 9,
    'p2' : 10,
    'n3' : 11,
    'p3' : 12,
    'n4' : 13,
    'p4' : 14,
    'n5' : 15,
    'p5' : 16,
    'n6' : 17,
    'p6' : 18
}

LINK = {
    'id': 0,
    'link_type': 1,
    'node_a' : 2,
    'node_b' : 3,
    'diameter' : 4,
    'area' : 5,
    'wave_speed' : 6,
    'ffactor' : 7,
    'length' : 8,
    'dx' : 9,
    'setting': 10
}

NODE_TYPES = {
    'None' : NULL,
    'Interior' : 0,
    # boundary nodes
    'Reservoir' : 1,
    'Junction' : 2,
    'Valve' : 3,
    'Pump' : 4
}

LINK_TYPES = {
    'None' : NULL,
    'Pipe': 0,
    'Valve': 1
}