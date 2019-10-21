import numpy as np
from phammer.simulation.util import imerge

def even(num_points, num_processors):
    p = np.ones(num_points)
    n = num_points // num_processors
    r = num_points % num_processors
    for i in range(num_processors):
        start = i*n
        end = start + n
        if i < r:
            start += i; end += i
        elif r > 0:
            start += r; end += r
        p[start:end+1] = i
    return p

def get_partition(processors, rank, where, ic, wn):

    # List of points needs to be sorted
    worker_points = np.where(processors == rank)[0]
    worker_points.sort()
    worker_pipes = where.points['to_pipes'][worker_points]
    diff = np.where(np.diff(worker_pipes) > 0)[0]
    extradiff = []

    if len(diff) > 0:
        if worker_points[diff[0]] != worker_points[0]:
            extradiff.append(0)
        if worker_points[diff[-1]]+1 != worker_points[-1]:
            extradiff.append(-1)
    else:
        extradiff.append(0)
        extradiff.append(-1)

    # d is the list of indexes in worker_points that correspond to
    # ghost nodes that are assigned to the processor with id == rank
    d = np.zeros(len(diff)*2 + len(extradiff), dtype=int)

    if len(extradiff) == 2:
        d[1:-1] = imerge(diff, diff+1)
        d[-1] = extradiff[-1]
    elif len(extradiff) == 1:
        if extradiff[0] == 0:
            d[1:] = imerge(diff, diff+1)
        else:
            d[0:-1] = imerge(diff, diff+1)
            d[-1] = extradiff[-1]
    elif len(diff) > 0:
        d[:] = imerge(diff, diff+1)
    elif len(diff) == 0:
        d[:] = extradiff

    points_idx = np.ones_like(worker_points).astype(bool)
    points_idx[d] = 0
    dependent = worker_points[d]
    # 1: uboundary, 0: dboundary
    dependent_type = np.isin(dependent, where.points['are_uboundaries'])
    points = list(worker_points[points_idx])

    boundaries_to_nodes = {
        point : where.pipes['to_nodes'][i]
        for i, point in enumerate(where.points['are_boundaries'])
    }

    nodes = []; node_points_list = []; node_points_context = [0]
    tanks = []; tanks_points_list = []
    reservoirs = []; reservoirs_points_list = []
    inline_valves = []; start_inline_valves = []; end_inline_valves = []
    inline_pumps = []; start_inline_pumps = [];  end_inline_pumps = []
    single_valves = []; single_valve_points = []
    single_pumps = []; single_pump_points = []

    # Determine extra data for dependent points
    visited_nodes = []

    for i, b in enumerate(dependent):
        if b in boundaries_to_nodes:
            # Boundary point
            node = boundaries_to_nodes[b]
            degree = where.nodes['to_points',][node]

            if node in visited_nodes:
                continue

            if degree == 1:
                nnode = wn.get_node(ic['node'].ival(node))
                links = wn.get_links_for_node(ic['node'].ival(node))

                if len(links) == 1:
                    l = wn.get_link(links[0])
                    snode = l.start_node
                    enode = l.end_node
                    if ic['pipe'].direction[l.name] < 0:
                        snode, enode = enode, snode

                    if nnode.node_type in ('Reservoir', 'Tank'):
                        if snode == node:
                            points.append(b+1)
                        elif enode == node:
                            points.append(b-1)
                        visited_nodes.append(node)
                        points.append(b)
                    if nnode.node_type == 'Tank':
                        tanks.append(node); tanks_points_list.append(b); continue
                    if nnode.node_type == 'Reservoir':
                        reservoirs.append(node); reservoirs_points_list.append(b); continue

                elif len(links) > 1:
                    l1 = wn.get_link(links[0])
                    l2 = wn.get_link(links[1])

                    nonpipe = l1 if l1.link_type.lower() != 'pipe' else l2

                    nonpipe_start = ic['node'].iloc(nonpipe.start_node.name)
                    nonpipe_end = ic['node'].iloc(nonpipe.end_node.name)
                    nonpipe_type = nonpipe.link_type.lower()
                    inline = sum(ic['node'].degree[[nonpipe_start, nonpipe_end]]) > 3
                    x = ic['node'].degree[[nonpipe_start, nonpipe_end]]

                    if inline:
                        if node == nonpipe_end and nonpipe_type == 'valve':
                            valve = np.where(where.points['end_inline_valve'] == b)[0][0]
                            processors[b] = processors[where.points['start_inline_valve'][valve]]
                            visited_nodes.append(node); points.append(b); continue
                        if node == nonpipe_end and nonpipe_type == 'pump':
                            pump = np.where(where.points['end_inline_pump'] == b)[0][0]
                            processors[b] = processors[where.points['start_inline_pump'][pump]]
                            visited_nodes.append(node); points.append(b); continue
                        if node == nonpipe_start and nonpipe_type == 'valve':
                            valve = np.where(where.points['start_inline_valve'] == b)[0][0]
                            valve_real = where.points['start_inline_valve',][valve]
                            point_end_valve = where.points['end_inline_valve'][valve]
                            pipe = wn.get_links_for_node(ic['node'].ival(ic['valve'].end_node[valve_real]))
                            pipe.remove(ic['valve'].ival(valve_real))
                            dpipe = ic['pipe'].iloc(pipe[0])
                            dpipe_points = where.points['are_boundaries'][2*dpipe:2*dpipe+2]
                            pad = 1 if dpipe_points[0] == point_end_valve else -1
                            visited_nodes.append(node)
                            points.append(point_end_valve+pad)
                            points.append(point_end_valve)
                            points.append(b)
                            processors[point_end_valve] = rank
                            start_inline_valves.append(b)
                            end_inline_valves.append(point_end_valve)
                            inline_valves.append(valve_real)
                            continue
                        if node == nonpipe_start and nonpipe_type == 'pump':
                            pump = np.where(where.points['start_inline_pump'] == b)[0][0]
                            pump_real = where.points['start_inline_pump',][pump]
                            point_end_pump = where.points['end_inline_pump'][pump]
                            pipe = wn.get_links_for_node(ic['node'].ival(ic['pump'].end_node[pump_real]))
                            pipe.remove(ic['pump'].ival(pump_real))
                            dpipe = ic['pipe'].iloc(pipe[0])
                            dpipe_points = where.points['are_boundaries'][2*dpipe:2*dpipe+2]
                            pad = 1 if dpipe_points[0] == point_end_pump else -1
                            points.append(point_end_pump+pad)
                            points.append(point_end_pump)
                            points.append(b)
                            processors[point_end_pump] = rank
                            visited_nodes.append(node)
                            start_inline_pumps.append(b)
                            end_inline_pumps.append(point_end_pump)
                            inline_pumps.append(pump_real)
                            continue
                    else:
                        if nonpipe_type == 'pump':
                            single_pump_points.append(b)
                            pump = np.where(where.points['are_single_pump'] == b)[0][0]
                            single_pumps.append(where.points['are_single_pump',][pump])
                        elif nonpipe_type == 'valve':
                            single_valve_points.append(b)
                            valve = np.where(where.points['are_single_valve'] == b)[0][0]
                            single_valves.append(where.points['are_single_valve',][valve])
                        if node == nonpipe_end:
                            points.append(b)
                            points.append(b-1)
                            visited_nodes.append(node)
                            continue
                        if node == nonpipe_start:
                            points.append(b)
                            points.append(b+1)
                            visited_nodes.append(node)
                            continue

            start = sum(where.nodes['to_points',][:node])
            end = start + where.nodes['to_points',][node]
            node_points = where.nodes['to_points'][start:end]
            processor_in_charge = min(processors[node_points])

            if processor_in_charge != rank:
                points.append(b)
                continue

            x = -1 if dependent_type[i] else 1

            if processors[b+x] != rank: # extra point needed
                points.append(b+x)
            if where.nodes['to_points',][node] == 1:
                points.append(b)
                visited_nodes.append(node)
                continue

            to_points_are_uboundaries = where.nodes['to_points_are_uboundaries'][start:end]
            slots = np.arange(len(node_points)*2)
            idx = slots[slots % 2 == 0] + to_points_are_uboundaries
            slots[idx] = node_points
            pad = 1 - 2*to_points_are_uboundaries
            slots[idx + pad] = node_points + pad
            points += list(slots)
            processors[node_points] = processor_in_charge
            nodes.append(node)
            node_points_list += list(node_points)
            node_points_context.append(len(node_points))
            visited_nodes.append(node)
        else:
            # Inner point
            node = None
            if b-1 in boundaries_to_nodes:
                node = boundaries_to_nodes[b-1]
            if b+1 in boundaries_to_nodes:
                node = boundaries_to_nodes[b+1]
            if not node is None:
                start = sum(where.nodes['to_points',][:node])
                end = start + where.nodes['to_points',][node]
                node_points = where.nodes['to_points'][start:end]
                processors[node_points] = min(processors[node_points])
            points.append(b-1); points.append(b); points.append(b+1)

    points = np.unique(points)
    points.sort()
    p = processors[points]
    rcv = np.where(p != rank)

    partition = {
        'points' : points,
        'nodes' : {
            'global_idx' : np.array(nodes, dtype = int),
            'points' : np.array(node_points_list, dtype = int),
            'context' : np.array(node_points_context, dtype = int),
        },
        'tanks' : {
            'global_idx' : np.array(tanks, dtype = int),
            'points' : np.array(tanks_points_list, dtype = int),
        },
        'reservoirs' : {
            'global_idx' : np.array(reservoirs, dtype = int),
            'points' : np.array(reservoirs_points_list, dtype = int),
        },
        'inline_valves' : {
            'global_idx' : np.array(inline_valves, dtype = int),
            'start_points' : np.array(start_inline_valves, dtype = int),
            'end_points' : np.array(end_inline_valves, dtype = int),
        },
        'inline_pumps' : {
            'global_idx' : np.array(inline_pumps, dtype = int),
            'start_points' : np.array(start_inline_pumps, dtype = int),
            'end_points' : np.array(end_inline_pumps, dtype = int),
        },
        'single_valves' : {
            'global_idx' : np.array(single_valves, dtype = int),
            'points' : np.array(single_valve_points, dtype = int),
        },
        'single_pumps' : {
            'global_idx' : np.array(single_pumps, dtype = int),
            'points' : np.array(single_pump_points, dtype = int),
        }
    }

    return partition, rcv