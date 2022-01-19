import numpy as np

from ptsnet.epanet.util import EN
from ptsnet.simulation.constants import TOL

class ModelError(Exception):
    pass

def check_compatibility(wn, ss):

    min_degree = min(wn.get_graph().degree)
    if min_degree[1] == 0:
        raise ModelError("node '%s' is isolated" % min_degree[0])

    not_inline_pumps = np.arange(wn.num_pumps)[~ss['pump'].is_inline]
    inline_pumps = np.arange(wn.num_pumps)[ss['pump'].is_inline]
    inline_valves = np.arange(wn.num_valves)[ss['valve'].is_inline]

    # Pumps can not have dead ends
    possible_dead_end = ss['node'].degree[ss['pump'].end_node] < 2
    if possible_dead_end.any():
        raise ModelError("there are pumps with an incompatible end node" % ss['node'].labels[possible_dead_end])

    # Valves can not have isolated start nodes
    possible_isolated_vstart = ss['node'].degree[ss['valve'].start_node] < 2
    if possible_isolated_vstart.any():
        raise ModelError("there are valves with an incompatible start node" % ss['node'].labels[possible_isolated_vstart])

    # Pumps can not have isolated start nodes that are not reservoirs
    incompatible_start_pnodes = (ss['node'].degree[ss['pump'].start_node[not_inline_pumps]] == 1) * \
        (ss['node'].degree[ss['pump'].end_node[not_inline_pumps]] > 1)

    incompatible_start_pnodes = ~np.isin(
        ss['node'].type[ss['pump'].start_node[not_inline_pumps][incompatible_start_pnodes]], (EN.TANK, EN.RESERVOIR,))
    incompatible_start_pnodes = ss['node'].labels[not_inline_pumps][incompatible_start_pnodes]

    if len(incompatible_start_pnodes) > 0:
        raise ModelError("start nodes of pumps are incompatible: \n%s" % incompatible_start_pnodes)

    # Nodes of non-pipe elements can not be general junctions
    start_valve_error = ss['node'].degree[ss['valve'].start_node] > 2
    start_pump_error = ss['node'].degree[ss['pump'].start_node] > 2
    end_valve_error = ss['node'].degree[ss['valve'].end_node] > 2
    end_pump_error = ss['node'].degree[ss['pump'].end_node] > 2
    if start_valve_error.any():
        raise ModelError(
            "start nodes of valves are connected to more than one pipe: \n%s" % \
                ss['node'].labels[ss['valve'].start_node][start_valve_error])
    if start_pump_error.any():
        raise ModelError(
            "start nodes of pumps are connected to more than one pipe: \n%s" % \
                ss['node'].labels[ss['pump'].start_node][start_pump_error])
    if end_valve_error.any():
        raise ModelError(
            "end nodes of valves are connected to more than one pipe: \n%s" % \
                ss['node'].labels[ss['valve'].end_node][end_valve_error])
    if end_pump_error.any():
        raise ModelError(
            "end nodes of pumps are connected to more than one pipe: \n%s" % \
                ss['node'].labels[ss['pump'].end_node][end_pump_error])

    # Reservoirs can not be at the end of a pump
    if (ss['node'].type[ss['pump'].end_node] == EN.RESERVOIR).any():
        raise ModelError("there is a pump with a reservoir at its end node")
    # Reservoirs can not be at the end of a valve
    if (ss['node'].type[ss['valve'].end_node] == EN.RESERVOIR).any():
        raise ModelError("there is a valvve with a reservoir at its end node")

    all_non_pipe_nodes = np.concatenate((
        ss['pump'].start_node,
        ss['pump'].end_node,
        ss['valve'].start_node,
        ss['valve'].end_node))

    all_non_pipe_nodes_nburst = np.concatenate((
        ss['pump'].start_node,
        ss['pump'].end_node[inline_pumps],
        ss['valve'].start_node,
        ss['valve'].end_node[inline_valves]))

    # Non-pipe elements can only be connected to pipes
    if len(all_non_pipe_nodes) != len(np.unique(all_non_pipe_nodes)):
        raise ModelError("there are non-pipe elements connected to each other")

    # No leaks/demands are allowed for nodes of non-pipe elements (except end-valve)
    leaking = ss['node'].leak_coefficient[all_non_pipe_nodes_nburst] > 0
    demanded = ss['node'].demand_coefficient[all_non_pipe_nodes_nburst] > 0
    if leaking.any():
        raise ModelError("there is a non-pipe element connected to a leaking node\n%s" % \
            str(ss['node'].labels[all_non_pipe_nodes_nburst][leaking]))
    if demanded.any():
        raise ModelError("there are non-pipe elements connected to a node with demand\n%s" \
            % str(ss['node'].labels[all_non_pipe_nodes_nburst][demanded]))

    # Demands cannot be negative
    negative_demands = ss['node'].labels[np.where(ss['node'].demand < 0)[0]]
    negative_demands = negative_demands[np.where(~np.isin(negative_demands, wn.reservoir_name_list + wn.tank_name_list))[0]]
    if len(negative_demands) > 0:
        raise ModelError("nodes ['" + "', '".join(negative_demands) + "'] have negative demands")

    # Valves cannot have zero head_loss if initial flow is not zero
    non_zero_flow_valves = np.where((ss['valve'].flowrate != 0) & (ss['valve'].head_loss < TOL))[0]
    if len(non_zero_flow_valves) > 0:
        raise ModelError("valves ['" + "', '".join(ss['valve'].labels[non_zero_flow_valves]) + "'] cannot have zero (near-zero) loss coefficient")