# Imports

import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from time import time

matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'
matplotlib.rcParams['font.size'] = 22

assert len(sys.argv) > 1
inpfile = '/home/gandresr/Documents/GitHub/ptsnet/ptsnet/examples/TNET3_HAMMER.inp'

if sys.argv[1] != 'plot':
    if sys.argv[1] == 'ptsnet':
        from ptsnet.simulation.sim import PTSNETSimulation
        from ptsnet.utils.io import get_example_path
    else:
        import tsnet

plot_type = ''
nodes_names = ['JUNCTION-16', 'JUNCTION-20', 'JUNCTION-30', 'JUNCTION-45', 'JUNCTION-90',]# 'JUNCTION-23', '416-B']
colors = ['#CCCCCC', '#999999', '#666666']
markers = ['*', 'x', '.', '', 'o']
global_wave_speed = 1200
global_dt = 0.005

if sys.argv[1] == 'plot':
    data_hammer = np.loadtxt('results/burst_hammer.txt', delimiter=',', skiprows=1)
    data_ptsnet = np.loadtxt('results/burst_ptsnet.txt', delimiter=',', skiprows=1)
    data_tsnet = np.loadtxt('results/burst_tsnet.txt', delimiter=',', skiprows=1)
    if plot_type == 'separate':
        for i in range(len(nodes_names)):
            fig = plt.figure(figsize=(14.5, 8)); fig.clf(); ax = plt.subplot(111)
            plt.title(f'Head Resuls of {nodes_names[i]} (Burst)')
            ax.plot(data_hammer[:,0], data_hammer[:,1+i], '--', linewidth=4, label='HAMMER', color='#5a8ce6')
            ax.plot(data_tsnet[:,0], data_tsnet[:,1+i], linewidth=3, label='TSNET', color=colors[1])
            ax.plot(data_ptsnet[:,0], data_ptsnet[:,1+i], linewidth=3, label='PTSNET', color=colors[2])
            plt.xlim(0, 20)
            plt.xlabel('Time [s]'); plt.ylabel('Head [m]')
            ax.grid()
            box = ax.get_position()
            ax.set_position([box.x0, box.y0 + box.height * 0.15, box.width, box.height * 0.9])
            ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), fancybox=True, shadow=True, ncol=5)
            plt.savefig(f'figures/burst_{nodes_names[i]}.pdf')
    else:
        for simulator in ('HAMMER', 'TSNET', 'PTSNET'):
            fig = plt.figure(figsize=(14.5, 8)); fig.clf(); ax = plt.subplot(111)
            plt.title(f'Head Resuls for {simulator} (Burst)')
            for i in range(len(nodes_names)):
                if simulator == 'HAMMER':
                    ax.plot(data_hammer[:,0], data_hammer[:,1+i], linewidth=4, label=nodes_names[i])
                elif simulator == 'TSNET':
                    ax.plot(data_tsnet[:,0], data_tsnet[:,1+i], linewidth=4, label=nodes_names[i])
                elif simulator == 'PTSNET':
                    ax.plot(data_ptsnet[:,0], data_ptsnet[:,1+i], linewidth=4, label=nodes_names[i])
                plt.xlim(0, 20)
                plt.xlabel('Time [s]'); plt.ylabel('Head [m]')
                ax.grid()
                box = ax.get_position()
                # ax.set_position([box.x0, box.y0 + box.height * 0.15, box.width, box.height * 0.9])
                ax.legend()#loc='upper center', bbox_to_anchor=(0.5, -0.15), fancybox=True, shadow=True, ncol=5)
            plt.savefig(f'figures/burst_{simulator}.pdf')
elif sys.argv[1] == 'ptsnet':
    sim_time_0 = time()
    sim = PTSNETSimulation(
        inpfile = get_example_path('TNET3_HAMMER'),
        settings = {
            'duration': 20,
            'time_step': global_dt,
            'period' : 0,
            'default_wave_speed' : global_wave_speed})
    sim.add_burst('JUNCTION-73', 0.02, 1, 2)
    sim.run()
    sim_time_1 = time()
    rslts = [sim['node'].head[node] for node in nodes_names]
    np.savetxt(
        'results/burst_ptsnet.txt',
        list(zip(sim['time'], *rslts)),
        delimiter=',',
        header='Time,'+','.join(nodes_names),
        comments='')
    print(f'Duration: {sim_time_1-sim_time_0} s', )
else:
    sim_time_0 = time()
    tm = tsnet.network.TransientModel(inpfile)
    tm.set_wavespeed(global_wave_speed) # m/s
    t0 = 0; tf = 20   # simulation period [s]
    tm.set_time(tf, global_dt)
    print(f"Time step: {tm.time_step} s")
    # Add burst
    ts = 1 # burst start time
    tc = 1 # time for burst to fully develop
    final_burst_coeff = 0.02 # final burst coeff [ m^3/s/(m H20)^(1/2)]
    tm.add_burst('JUNCTION-73', ts, tc, final_burst_coeff)
    tm = tsnet.simulation.Initializer(tm, t0, 'DD')
    result_obj = 'Tnet3' # name of the object for saving simulation results
    tm = tsnet.simulation.MOCSimulator(tm,result_obj)
    sim_time_1 = time()
    tt = tm.simulation_timestamps
    rslts = []
    for node in nodes_names:
        nd = tm.get_node(node)
        rslts.append(nd.head)
    np.savetxt(
        'results/burst_tsnet.txt',
        list(zip(tt, *rslts)),
        delimiter=',',
        header='Time,'+','.join(nodes_names),
        comments='')
    print(f'Duration: {sim_time_1-sim_time_0} s')