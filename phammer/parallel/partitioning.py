import numpy as np
from phammer.simulation.util import imerge

def even(N, k):
    p = np.ones(N); n = N // k; r = N % k
    for i in range(k):
        start = i*n
        end = start + n
        if i < r:
            start += i; end += i
        elif r > 0:
            start += r; end += r
        p[start:end+1] = i
    return p

def get_points(processors, N, k, where, rank):
    # List of points needs to be sorted
    worker_points = np.where(processors == rank)[0]
    worker_points.sort()
    worker_pipes = where.points['to_pipes'][worker_points]
    diff = np.where(np.diff(worker_pipes) > 0)[0]
    extradiff = []

    if worker_points[diff[0]] != worker_points[0]:
        extradiff.append(0)
    if worker_points[diff[-1]]+1 != worker_points[-1]:
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
    else:
        d[:] = imerge(diff, diff+1)

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
        try:
            # Boundary point
            node = boundaries_to_nodes[b]
            degree = where.nodes['to_points',][node]

            if node in visited_nodes:
                continue
            if degree == 1:
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
        except:
            # Inner point
            points.append(b-1)
            points.append(b)
            points.append(b+1)

    points = np.unique(points)
    points.sort()
    return points