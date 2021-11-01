import numpy as np
import networkx as nx
import nxmetis as nxm

from collections import defaultdict as ddict
from ptsnet.utils.data import imerge
from pkg_resources import resource_filename

def even(sim, num_processors):
    processors = np.ones(sim.num_points, dtype = int)
    n = sim.num_points // num_processors
    r = sim.num_points % num_processors
    for i in range(num_processors):
        start = i*n
        end = start + n
        if i < r:
            start += i; end += i
        elif r > 0:
            start += r; end += r
        processors[start:end+1] = i
    return processors

def bisection(sim, num_processors):
    G = _get_numerical_grid(sim)
    num_cuts, partitioning = nxm.partition(G, num_processors)
    processors = -np.ones(sim.num_points)
    for p, partition in enumerate(partitioning):
        for node in partition:
            node_type = G.nodes[node]['node_type']
            if node_type == 'contracted':
                expanded_node = G.nodes[node]['expanded_node']
                if expanded_node:
                    processors[sim.get_node_points(expanded_node[0])] = p
                    processors[sim.get_node_points(expanded_node[1])] = p
            elif node_type == 'interior':
                processors[node] = p
            elif node_type == 'junction':
                processors[sim.get_node_points(node)] = p
    return processors

def _get_numerical_grid(sim):
    G = nx.Graph()
    contracted_nodes = {}
    for l in sim.wn.links:
        link = sim.wn.links[l]
        ltype = link.link_type.lower()
        if ltype in ('valve', 'pump'):
            n1 = sim.ic['node'].labels[sim.ic[ltype].start_node[l]]
            n2 = sim.ic['node'].labels[sim.ic[ltype].end_node[l]]
            contracted_nodes[n1] = n1
            contracted_nodes[n2] = n1
        elif ltype == 'pipe':
            link = sim.wn.links[l]
            ilink = sim.ic['pipe'].lloc(l)
            N = int(sim.ic['pipe'].segments[l])
            n1 = sim.ic['node'].labels[sim.ic['pipe'].start_node[l]]
            n2 = sim.ic['node'].labels[sim.ic['pipe'].end_node[l]]
            p1 = sim.where.points['are_boundaries'][ilink*2]
            p2 = sim.where.points['are_boundaries'][ilink*2+1]
            if n1 in contracted_nodes: n1 = contracted_nodes[n1]
            if n2 in contracted_nodes: n2 = contracted_nodes[n2]

            expanded_n1 = []
            if n1 in contracted_nodes:
                n1_type = 'contracted'
                if contracted_nodes[n1] != n1:
                    expanded_n1 = [n1, contracted_nodes[n1]]
            else:
                n1_type = 'junction'

            expanded_n2 = []
            if n2 in contracted_nodes:
                n2_type = 'contracted'
                if contracted_nodes[n2] != n2:
                    expanded_n2 = [n2, contracted_nodes[n2]]
            else:
                n2_type = 'junction'

            if not n1 in G: G.add_node(n1, node_type=n1_type, expanded_node=expanded_n1)
            if not n2 in G: G.add_node(n2, node_type=n2_type, expanded_node=expanded_n2)
            k = p1+1;  G.add_node(k, node_type='interior', expanded_node=[])

            G.add_edge(n1, k)
            for j in range(N-2):
                G.add_node(k+1, node_type='interior', expanded_node=[])
                G.add_edge(k, k+1)
                k += 1
            G.add_edge(k, n2)
    return G

def _get_ghost_points(worker_points, worker_pipes):
    worker_points.sort()
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
    ghosts = np.zeros(len(diff)*2 + len(extradiff), dtype=int)

    if len(extradiff) == 2:
        ghosts[1:-1] = imerge(diff, diff+1)
        ghosts[-1] = extradiff[-1]
    elif len(extradiff) == 1:
        if extradiff[0] == 0:
            ghosts[1:] = imerge(diff, diff+1)
        else:
            ghosts[0:-1] = imerge(diff, diff+1)
            ghosts[-1] = extradiff[-1]
    elif len(diff) > 0:
        ghosts[:] = imerge(diff, diff+1)
    elif len(diff) == 0:
        ghosts[:] = extradiff

    return ghosts

def get_partition(processors, rank, where, ic, wn, num_processors, inpfile):

    # List of points needs to be sorted
    worker_points = np.where(processors == rank)[0]
    worker_pipes = where.points['to_pipes'][worker_points]
    ghosts = _get_ghost_points(worker_points, worker_pipes)
    points_idx = np.ones_like(worker_points).astype(bool)
    points_idx[ghosts] = 0
    dependent = worker_points[ghosts]
    dependent_type = np.isin(dependent, where.points['are_uboundaries']) # 1: uboundary, 0: dboundary
    points = list(worker_points[points_idx])

    boundaries_to_nodes = {
        point : where.pipes['to_nodes'][i]
        for i, point in enumerate(where.points['are_boundaries'])
    }

    nodes = []; node_points_list = []; node_points_context = [0]
    tanks = []; tanks_points_list = []
    reservoirs = []; reservoirs_points_list = []
    inline_valves = []; start_valves = []; end_valves = []
    inline_pumps = []; start_pumps = [];  end_pumps = []
    single_valves = []; single_valve_points = []
    single_pumps = []; single_pump_points = []

    visited_nodes = []

    start_points = np.cumsum(where.nodes['to_points',])

    # Determine extra data for dependent points
    for i, b in enumerate(dependent):
        if b in boundaries_to_nodes: # Boundary point
            node = boundaries_to_nodes[b]
            degree = where.nodes['to_points',][node]

            nnode = wn.get_node(ic['node'].ilabel(node))
            l = wn.get_link(ic['pipe'].ilabel(where.points['to_pipes'][b]))
            snode = ic['pipe'].start_node[l.name]
            enode = ic['pipe'].end_node[l.name]

            if nnode.node_type in ('Tank', 'Reservoir'):
                if snode == node:
                    points.append(b+1)
                elif enode == node:
                    points.append(b-1)
                visited_nodes.append(node)
                points.append(b)
            if nnode.node_type == 'Tank':
                tanks.append(node)
                tanks_points_list.append(b)
                continue
            if nnode.node_type == 'Reservoir':
                reservoirs.append(node)
                reservoirs_points_list.append(b)
                continue

            if node in visited_nodes:
                continue

            if degree == 1:
                links = wn.get_links_for_node(nnode.name)
                if len(links) > 1:
                    l1 = wn.get_link(links[0])
                    l2 = wn.get_link(links[1])
                    nonpipe = l1 if l1.link_type.lower() != 'pipe' else l2
                    nonpipe_type = nonpipe.link_type.lower()
                    non_pipe_idx = ic[nonpipe_type].lloc(nonpipe.name)
                    nonpipe_start = ic[nonpipe_type].start_node[non_pipe_idx]
                    nonpipe_end = ic[nonpipe_type].end_node[non_pipe_idx]
                    is_inline = ic[nonpipe_type].is_inline[non_pipe_idx]

                    start_deg, end_deg = ic['node'].degree[[nonpipe_start, nonpipe_end]]
                    nonpipe_start_point = None
                    nonpipe_end_point = None

                    # Get upstream and downstream pipes
                    if start_deg == 2:
                        start_links = wn.get_links_for_node(ic['node'].ilabel(nonpipe_start))
                        start_links.remove(nonpipe.name)
                        start_pipe_idx = ic['pipe'].lloc(start_links[0])
                        if nonpipe_start == ic['pipe'].start_node[start_pipe_idx]:
                            npipe_spoint = 2*start_pipe_idx
                        elif nonpipe_start == ic['pipe'].end_node[start_pipe_idx]:
                            npipe_spoint = 2*start_pipe_idx + 1
                        nonpipe_start_point = where.points['are_boundaries'][npipe_spoint]
                    if end_deg == 2:
                        end_links = wn.get_links_for_node(ic['node'].ilabel(nonpipe_end))
                        end_links.remove(nonpipe.name)
                        end_pipe_idx = ic['pipe'].lloc(end_links[0])
                        if nonpipe_end == ic['pipe'].start_node[end_pipe_idx]:
                            npipe_epoint = 2*end_pipe_idx
                        elif nonpipe_end == ic['pipe'].end_node[end_pipe_idx]:
                            npipe_epoint = 2*end_pipe_idx + 1
                        nonpipe_end_point = where.points['are_boundaries'][npipe_epoint]
                    if is_inline:
                        nonpipe_points = [nonpipe_start_point, nonpipe_end_point]
                        processor_in_charge = min(processors[nonpipe_points])
                        processors[nonpipe_points] = processor_in_charge
                        points.append(b)
                        if processor_in_charge != rank:
                            visited_nodes.append(node)
                            continue
                        else:
                            points.append(nonpipe_start_point)
                            points.append(nonpipe_end_point)
                            if processors[nonpipe_start_point - 1] != rank:
                                points.append(nonpipe_start_point-1)
                            if processors[nonpipe_end_point + 1] != rank:
                                points.append(nonpipe_end_point+1)
                            if nonpipe_type == 'valve':
                                loc = ic['valve'].lloc(nonpipe.name)
                                if not loc in inline_valves:
                                    inline_valves.append(loc)
                                    start_valves.append(nonpipe_start_point)
                                    end_valves.append(nonpipe_end_point)
                            elif nonpipe_type == 'pump':
                                loc = ic['pump'].lloc(nonpipe.name)
                                if not loc in inline_pumps:
                                    inline_pumps.append(loc)
                                    start_pumps.append(nonpipe_start_point)
                                    end_pumps.append(nonpipe_end_point)
                            visited_nodes.append(node)
                            continue
                        continue
                    else:
                        points.append(b)
                        visited_nodes.append(node)
                        if nonpipe_type == 'valve':
                            if processors[b - 1] != rank:
                                points.append(b - 1)
                            single_valves.append(ic['valve'].lloc(nonpipe.name))
                            single_valve_points.append(b)
                            continue
                        if nonpipe_type == 'pump':
                            if processors[b + 1] != rank:
                                points.append(b + 1)
                            single_pumps.append(ic['pump'].lloc(nonpipe.name))
                            single_pump_points.append(b)
                            continue
                        continue

            start = 0 if node == 0 else start_points[node-1]
            end = start + where.nodes['to_points',][node]
            node_points = where.nodes['to_points'][start:end]
            processor_in_charge = min(processors[node_points])
            processors[node_points] = processor_in_charge

            if processor_in_charge != rank:
                if dependent_type[i] == 1: # uboundary
                    if processors[b-1] == rank:
                        points.append(b)
                elif dependent_type[i] == 0: # dboundary
                    if processors[b+1] == rank:
                        points.append(b)
                continue

            x = -1 if dependent_type[i] else 1

            if processors[b+x] != rank: # extra point needed
                points.append(b+x)

            to_points_are_uboundaries = where.nodes['to_points_are_uboundaries'][start:end]
            slots = np.arange(len(node_points)*2)
            idx = slots[slots % 2 == 0] + to_points_are_uboundaries
            slots[idx] = node_points
            pad = 1 - 2*to_points_are_uboundaries
            slots[idx + pad] = node_points + pad
            points += list(slots)
            nodes.append(node)
            node_points_list += list(node_points)
            node_points_context.append(len(node_points))
            visited_nodes.append(node)
        else:
            # Inner point
            node = None
            if b-1 in boundaries_to_nodes:
                node = boundaries_to_nodes[b-1]
                start = 0 if node == 0 else start_points[node-1]
                end = start + where.nodes['to_points',][node]
                node_points = where.nodes['to_points'][start:end]
                processor_in_charge = min(processors[node_points])
                processors[node_points] = processor_in_charge
            if b+1 in boundaries_to_nodes:
                node = boundaries_to_nodes[b+1]
                start = 0 if node == 0 else start_points[node-1]
                end = start + where.nodes['to_points',][node]
                node_points = where.nodes['to_points'][start:end]
                processor_in_charge = min(processors[node_points])
                processors[node_points] = processor_in_charge
            points.append(b-1); points.append(b); points.append(b+1)

    points = np.unique(points)
    points.sort()

    if len(points) == 0:
        partition = None
    else:
        partition = {
            'points' : {
                'global_idx' : points,
                'local_idx' : {p : i for i, p in enumerate(points)}
            },
            'nodes' : {
                'global_idx' : np.array(nodes).astype(int),
                'points' : np.array(node_points_list).astype(int),
                'context' : np.array(node_points_context).astype(int),
            },
            'tanks' : {
                'global_idx' : np.array(tanks).astype(int),
                'points' : np.array(tanks_points_list).astype(int),
            },
            'reservoirs' : {
                'global_idx' : np.array(reservoirs).astype(int),
                'points' : np.array(reservoirs_points_list).astype(int),
            },
            'inline_valves' : {
                'global_idx' : np.array(inline_valves).astype(int),
                'start_points' : np.array(start_valves).astype(int),
                'end_points' : np.array(end_valves).astype(int),
            },
            'inline_pumps' : {
                'global_idx' : np.array(inline_pumps).astype(int),
                'start_points' : np.array(start_pumps).astype(int),
                'end_points' : np.array(end_pumps).astype(int),
            },
            'single_valves' : {
                'global_idx' : np.array(single_valves).astype(int),
                'points' : np.array(single_valve_points).astype(int),
            },
            'single_pumps' : {
                'global_idx' : np.array(single_pumps).astype(int),
                'points' : np.array(single_pump_points).astype(int),
            }
        }

    return partition