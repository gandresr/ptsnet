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

def get_points(processors, rank, where, ic, wn):

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
                if b in where.points['end_inline_valve']:
                    valve = np.where(where.points['end_inline_valve'] == b)[0][0]
                    point_start_valve = where.points['start_inline_valve'][valve]
                    processors[b] = processors[point_start_valve]
                    visited_nodes.append(node); points.append(b)
                    continue
                if b in where.points['end_inline_pump']:
                    pump = np.where(where.points['end_inline_pump'] == b)[0][0]
                    point_start_pump = where.points['start_inline_pump'][pump]
                    processors[b] = processors[point_start_pump]
                    visited_nodes.append(node); points.append(b)
                    continue
                if b in where.points['start_inline_valve']:
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
                    continue
                if b in where.points['start_inline_pump']:
                    pump = np.where(where.points['start_inline_pump'] == b)[0][0]
                    pump_real = where.points['start_inline_pump',][pump]
                    point_end_pump = where.points['end_inline_pump'][pump]
                    pipe = wn.get_links_for_node(ic['node'].ival(ic['pump'].end_node[pump_real]))
                    pipe.remove(ic['pump'].ival(pump_real))
                    dpipe = ic['pipe'].iloc(pipe[0])
                    dpipe_points = where.points['are_boundaries'][2*dpipe:2*dpipe+2]
                    pad = 1 if dpipe_points[0] == point_end_pump else -1
                    visited_nodes.append(node)
                    points.append(point_end_pump+pad)
                    points.append(point_end_pump)
                    points.append(b)
                    processors[point_end_pump] = rank
                    continue

                visited_nodes.append(node)
                points.append(b)
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

    return points, rcv