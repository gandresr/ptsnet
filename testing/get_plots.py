import os
import pickle
import matplotlib.pyplot as plt
import numpy as np
folders = os.listdir('results/BWSN_F')
p = []
get_partition = []
check_processor_innactivity = []
create_selectors = []
comm_graph = []
allocate_memory = []
load_initial_conditions = []
total_sim_time = []
run_step = []
run_interior_step = []
run_boundary_step = []
run_valve_step = []
run_pump_step = []
store_results = []
barrier = []
exchange_data = []

for folder in folders:
    if '.' in folder: continue
    with open('results/BWSN_F/' + folder + '/summary.dat', 'rb') as f:
        data = pickle.load(f)
    p.append(int(folder[folder.find('_') + 1:]))
    get_partition.append(data['get_partition'][-1])
    check_processor_innactivity.append(data['check_processor_innactivity'][-1])
    create_selectors.append(data['_create_selectors'][-1])
    comm_graph.append(data['_define_dist_graph_comm'][-1])
    allocate_memory.append(data['_allocate_memory'][-1])
    load_initial_conditions.append(data['_load_initial_conditions'][-1])
    total_sim_time.append(data['total_sim_time'][-1])
    run_step.append(data['run_step'][-1])
    run_interior_step.append(data['run_interior_step'][-1])
    run_boundary_step.append(data['run_boundary_step'][-1])
    run_valve_step.append(data['run_valve_step'][-1])
    run_pump_step.append(data['run_pump_step'][-1])
    store_results.append(data['store_results'][-1])
    if p[-1] > 1:
        barrier.append(data['barrier'][-1])
        exchange_data.append(data['exchange_data'][-1])
    else:
        barrier.append(0)
        exchange_data.append(0)

plt.figure()
temp = np.argsort(p)
ranks = np.empty_like(temp)
ranks[temp] = np.arange(len(p))
X = ranks
p1 = plt.bar(X, get_partition, width = 0.5)
p2 = plt.bar(X, check_processor_innactivity, bottom = get_partition, width = 0.5)
bars1 = np.add(get_partition, check_processor_innactivity)
p3 = plt.bar(X, create_selectors, bottom = bars1, width = 0.5)
bars2 = np.add(bars1, create_selectors)
p4 = plt.bar(X, comm_graph, bottom = bars2, width = 0.5)
bars3 = np.add(bars2, comm_graph)
p5 = plt.bar(X, allocate_memory, bottom = bars3, width = 0.5)
bars4 = np.add(bars3, allocate_memory)
p6 = plt.bar(X, load_initial_conditions, bottom = bars4, width = 0.5)
bars5 = np.add(bars4, load_initial_conditions)

p7 = plt.bar(X, np.array(run_interior_step)*850, bottom = bars5, width = 0.5)
bars6 = np.add(bars5, np.array(run_interior_step)*850)
p8 = plt.bar(X, np.array(run_boundary_step)*850, bottom = bars6, width = 0.5)
bars7 = np.add(bars6, np.array(run_boundary_step)*850)
p9 = plt.bar(X, np.array(run_valve_step)*850, bottom = bars7, width = 0.5)
bars8 = np.add(bars7, np.array(run_valve_step)*850)
p10 = plt.bar(X, np.array(run_pump_step)*850, bottom = bars8, width = 0.5)
bars9 = np.add(bars8, np.array(run_pump_step)*850)
p11 = plt.bar(X, np.array(store_results)*850, bottom = bars9, width = 0.5)
bars10 = np.add(bars9, np.array(store_results)*850)
p12 = plt.bar(X, np.array(barrier)*850*2+np.array(exchange_data)*850, bottom = bars10, width = 0.5)
plt.xticks(np.arange(len(p)), list(map(str, np.sort(p))))

plt.legend(
    (p1[0], p2[0], p3[0], p4[0], p5[0], p6[0], p7[0], p8[0], p9[0], p10[0], p11[0], p12[0]),
    (
        'get_partition',
        'check_processor_innactivity',
        'create_selectors',
        'comm_graph',
        'allocate_memory',
        'load_initial_conditions',
        'run_interior_step',
        'run_boundary_step',
        'run_valve_step',
        'run_pump_step',
        'store_results',
        'processor_communication'))
plt.xlabel('Number of processors')
plt.ylabel('Time [s]')
plt.title('Simulation times by subprocess')
plt.grid(True)
plt.show()