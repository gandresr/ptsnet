from mpi4py import MPI
import wntr
import numpy as np
import pickle
from ptsnet.utils.io import get_root_path

comm = MPI.COMM_WORLD
rank = comm.rank
inpfile = get_root_path() + 'example_files/BWSN_F.inp'
wn = wntr.network.WaterNetworkModel(inpfile)

node_results = [[],[]]
pipe_start_results = [[],[]]
pipe_end_results = [[],[]]

if rank > 0:
    for i, r in enumerate((rank, rank+1,)):
        for j in range(r):
            fname = 'rank_{r}/{i}.pickle'.format(r = r, i = i)
            with open(fname, 'rb') as f:
                data = pickle.load(f)
                node_results[i].append(data['node_results'])
                pipe_start_results[i].append(data['pipe_start_results'])
                pipe_end_results[i].append(data['pipe_end_results'])

    for node in wn.node_name_list:
        nr_0 = None; nr_1 = None
        for nr in node_results[0]:
            try:
                nr_0 = nr.head[node]
            except:
                continue
        for nr in node_results[1]:
            try:
                nr_1 = nr.head[node]
            except:
                continue
        if (not nr_0 is None) and (not nr_1 is None):
            if not np.all(nr_0 == nr_1):
                raise RuntimeError('Results for node %s do not coincide for processors %d and %d' % (node, rank, rank+1))
        else:
            raise RuntimeError('Results not found for node %s' % node)

    for pipe in wn.pipe_name_list:

        psr_0 = None; psr_1 = None
        per_0 = None; per_1 = None

        for pr in pipe_start_results[0]:
            try:
                psr_0 = pr.flowrate[pipe]
            except:
                continue
        for pr in pipe_end_results[0]:
            try:
                per_0 = pr.flowrate[pipe]
            except:
                continue
        for pr in pipe_start_results[1]:
            try:
                psr_1 = pr.flowrate[pipe]
            except:
                continue
        for pr in pipe_end_results[1]:
            try:
                per_1 = pr.flowrate[pipe]
            except:
                continue
        if (not psr_0 is None) and (not psr_1 is None):
            if not np.all(psr_0 == psr_1):
                raise RuntimeError('Start results for pipe %s do not coincide for processors %d and %d' % (pipe, rank, rank+1))
            if not np.all(per_0 == per_1):
                raise RuntimeError('End results for pipe %s do not coincide for processors %d and %d' % (pipe, rank, rank+1))
        else:
            raise RuntimeError('Results not found for pipe %s' % pipe)

    print(True, rank, rank+1)
