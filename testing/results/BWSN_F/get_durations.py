import os

import numpy as np
import pickle as pk
from collections import defaultdict as ddict
from phammer.arrays import Table, Table2D
import ntpath

folders = [x[0] for x in os.walk(os.getcwd())]

PROFILER_INIT_DATA = {
    'get_partition' : np.float,
    'check_processor_innactivity' : np.float,
    '_create_selectors' : np.float,
    '_define_dist_graph_comm' : np.float,
    '_allocate_memory' : np.float,
    '_load_initial_conditions' : np.float,
    'total_sim_time' : np.float,
}

PROFILER_STEP_DATA = {
    'run_step' : np.float,
    'run_interior_step' : np.float,
    'run_boundary_step' : np.float,
    'run_valve_step' : np.float,
    'run_pump_step' : np.float,
    'store_results' : np.float,
    'barrier1' : np.float,
    'exchange_data' : np.float,
    'barrier2' : np.float,
}

parent_path = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
parent_folder = ntpath.basename(parent_path)
num_steps = int(parent_folder[parent_folder.rfind('_')+1:])

for folder in folders:

    full_data = ddict(list)
    num_processors = int(folder[folder.rfind('_')+1:])
    step_data = Table2D(PROFILER_STEP_DATA, num_processors, 2*num_steps)
    profiler_start_times = Table(PROFILER_INIT_DATA, num_processors)
    profiler_end_times = Table(PROFILER_INIT_DATA, num_processors)

    for r in np.arange(1,num_processors+1):
        with open(folder + '/data_' + str(r) + '.pickle', 'rb') as f:
            data = pk.load(f)
        jobs = data['sim_times'].jobs
        step_data.run_step[r][:] = data['run_step'].time_stamps
        step_data.run_interior_step[r][:] = data['run_interior_step'].time_stamps
        step_data.run_boundary_step[r][:] = data['run_boundary_step'].time_stamps
        step_data.run_valve_step[r][:] = data['run_valve_step'].time_stamps
        step_data.run_pump_step[r][:] = data['run_pump_step'].time_stamps
        step_data.store_results[r][:] = data['store_results'].time_stamps
        step_data.barrier1[r][:] = data['barrier1'].time_stamps
        step_data.exchange_data[r][:] = data['exchange_data'].time_stamps
        step_data.barrier2[r][:] = data['barrier2'].time_stamps
        profiler_start_times.get_partition[r] = data['get_partition'][0]; print(len(data['get_partition']) == 2)
        profiler_start_times.check_processor_innactivity[r] = data['check_processor_innactivity'][0]; print(len(data['check_processor_innactivity']) == 2)
        profiler_start_times._create_selectors[r] = data['_create_selectors'][0]; print(len(data['_create_selectors']) == 2)
        profiler_start_times._define_dist_graph_comm[r] = data['_define_dist_graph_comm'][0]; print(len(data['_define_dist_graph_comm']) == 2)
        profiler_start_times._allocate_memory[r] = data['_allocate_memory'][0]; print(len(data['_allocate_memory']) == 2)
        profiler_start_times._load_initial_conditions[r] = data['_load_initial_conditions'][0]; print(len(data['_load_initial_conditions']) == 2)
        profiler_start_times.total_sim_time[r] = data['total_sim_time'][0]; print(len(data['total_sim_time']) == 2)
        profiler_end_times.get_partition[r] = data['get_partition'][1]
        profiler_end_times.check_processor_innactivity[r] = data['check_processor_innactivity'][1]
        profiler_end_times._create_selectors[r] = data['_create_selectors'][1]
        profiler_end_times._define_dist_graph_comm[r] = data['_define_dist_graph_comm'][1]
        profiler_end_times._allocate_memory[r] = data['_allocate_memory'][1]
        profiler_end_times._load_initial_conditions[r] = data['_load_initial_conditions'][1]
        profiler_end_times.total_sim_time[r] = data['total_sim_time'][1]

    durations = {}
    pmax = np.argmax(profiler_end_times._load_initial_conditions)

    for property in profiler_end_times.properties:
        durations[property] = profiler_end_times.__dict__[property][pmax] - profiler_start_times.__dict__[property][pmax]

    total_times = ddict(int)
    for i in range(num_steps):
        mmax = -1e20; xmax = -1e20
        pmax = None; pmax_exchange = None
        for j in range(1, num_processors, 2):
            nval = step_data.run_step[j][i]
            xval = step_data.exchange_data[j][i]
            if nval > nmax:
                nmax = nval; pmax = j
            if xval > xmax:
                xmax = xval; pmax = j
        for property in step_data.properties:
            total_times[property] += step_data.__dict__[property][pmax][i]

    for k in total_times:
        durations[k] = total_times[k]

    with open(folder + '/durations.dat', 'wb') as fs:
        pk.dump(durations, fs)