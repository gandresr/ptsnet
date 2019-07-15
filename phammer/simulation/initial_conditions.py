import numpy as np
from phammer.mesh.constants import *

def get_initial_conditions(mesh):
    Q0 = np.zeros(mesh.num_points, dtype = np.float)
    H0 = np.zeros(mesh.num_points, dtype = np.float)

    for i in range(mesh.num_points):
        link_id = mesh.properties['int']['points'].link_id[i]
        link_name = mesh.wn.link_name_list[link_id]

        link = mesh.wn.get_link(link_name)

        Q0[i] = float(mesh.steady_state_sim.link['flowrate'][link_name])
        start_node_name = link.start_node_name
        end_node_name = link.end_node_name
        k = mesh.properties['int']['points'].subindex[i]

        if k == 0:
            H0[i] = float(mesh.steady_state_sim.node['head'][start_node_name])
        elif k == mesh.segments[link_name]:
            H0[i] = float(mesh.steady_state_sim.node['head'][end_node_name])
        else:
            head_1 = float(mesh.steady_state_sim.node['head'][start_node_name])
            head_2 = float(mesh.steady_state_sim.node['head'][end_node_name])
            dx = k * link.length / mesh.segments[link_name]
            if head_1 > head_2:
                hl = head_1 - head_2
                H0[i] = head_1 - hl*dx/link.length
            else:
                hl = head_2 - head_1
                H0[i] = head_1 + hl*dx/link.length
    return Q0, H0