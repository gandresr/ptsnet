# import matplotlib.pyplot as plt
import numpy as np
# import pandas as pd
import wntr
import os
import pickle
import matplotlib.pyplot as plt
import ntpath
from phammer.simulation.sim import HammerSimulation
from time import time
from phammer.utils.io import get_root_path

ROOT = get_root_path()
duration = 5; time_step = 1
inpfile = os.path.join(ROOT, os.pardir, 'example_files', 'BWSN_F.inp')

sim = HammerSimulation(
    inpfile,
    {
        'time_step' : time_step,
        'duration' : duration,
        'skip_compatibility_check' : False,
        'warnings_on' : False,
        'show_progress' : True,
    },
    period = 219,
    default_wave_speed = 1000)

sim.add_curve('V_BUTTERFLY', 'valve',
    [1, 0.8, 0.6, 0.4, 0.2, 0],
    [0.067, 0.044, 0.024, 0.011, 0.004, 0.   ])

valves = sim.wn.valve_name_list
pumps = sim.wn.pump_name_list
sim.assign_curve_to('V_BUTTERFLY', valves)

SR = int(1//sim.settings.time_step)
for valve in valves:
    sim.define_valve_settings(valve, np.linspace(0, 1, SR), np.linspace(1, 0, SR))

for pump in pumps:
    sim.define_pump_settings(pump, np.linspace(0, 1, SR), np.linspace(1, 0, SR))

sim.initialize()

sim.worker.profiler.start('total_sim_time')
while not sim.is_over:
    sim.run_step()
sim.worker.profiler.stop('total_sim_time')

# fname = ntpath.basename(inpfile)
# fname = fname[:fname.find('.')]
# os.makedirs(ROOT + 'testing/results/{fname}_{time_steps}/rank_{comm_size}'.format(
#     fname = fname,
#     comm_size = str(sim.comm.size),
#     time_steps = sim.settings.time_steps), exist_ok = True)
# tt = np.linspace(0, duration, sim.settings.time_steps)
# with open(ROOT + 'testing/results/BWSN_F_{num_steps}/rank_{comm_size}/{rank}.pickle'.format(
#     comm_size = str(sim.comm.size), rank = str(sim.rank), num_steps = sim.settings.time_steps), 'wb') as f:
#     pickle.dump({
#         'num_points_global' : sim.num_points,
#         'num_points_worker' : sim.worker.num_points,
#         'wave_speeds' : sim.ic['pipe'].wave_speed,
#         'duration' : duration,
#         'time_step' : sim.settings.time_step,
#         'time_steps' : sim.settings.time_steps,
#         'tt' : tt,
#         'node_results' : sim.worker.node_results,
#         'pipe_start_results' : sim.worker.pipe_start_results,
#         'pipe_end_results' : sim.worker.pipe_end_results,
#         'sim_times' : sim.worker.profiler,
#     }, f)

# # head = pd.DataFrame(sim.node_results.head.T, columns = sim.node_results.labels)
# # in_flow = pd.DataFrame(sim.pipe_results.inflow.T, columns = sim.pipe_results.labels)
# # wntr.graphics.network.network_animation(sim.wn, node_attribute = head, link_attribute = in_flow)
# # plt.show()
