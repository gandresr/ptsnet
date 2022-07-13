## Pump shut-off

import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from time import time

matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'
matplotlib.rcParams['font.size'] = 22

assert len(sys.argv) > 1
inpfile = '/home/gandresr/Documents/GitHub/ptsnet/ptsnet/examples/TNET3.inp'

if sys.argv[1] != 'plot':
    if sys.argv[1] == 'ptsnet':
        from ptsnet.simulation.sim import PTSNETSimulation
        from ptsnet.utils.io import get_example_path
    else:
        import tsnet

plot_type = ''
nodes_names = ['217-B','JUNCTION-34', 'JUNCTION-30']#, 'JUNCTION-45', 'JUNCTION-90',]# 'JUNCTION-23', '416-B']
colors = ['#CCCCCC', '#999999', '#666666']
markers = ['*', 'x', '.', '', 'o']
global_wave_speed = 1200
global_dt = 0.005
grayscale = ['#007aff', '#4cd964', '#ff9500', '#ff3b30', '#5856d6']
# grayscale = ['#000000', '#333333', '#666666', '#999999', '#cccccc'] # bw
lstyle = ['-','--', '-', '-', '-']*5#, ':', '-.', ':', '-']
markers = ['', '', '', '', '']
alphas = [0.7, 1,1, 1, 1]

if sys.argv[1] == 'plot':
    data_hammer = np.loadtxt('results/pump_hammer.txt', delimiter=',', skiprows=1)
    data_ptsnet = np.loadtxt('results/pump_ptsnet.txt', delimiter=',', skiprows=1)
    data_tsnet = np.loadtxt('results/pump_tsnet.txt', delimiter=',', skiprows=1)
    if plot_type == 'separate':
        for i in range(len(nodes_names)):
            fig = plt.figure(figsize=(14.5, 8)); fig.clf(); ax = plt.subplot(111)
            plt.title(f'Head Resuls of {nodes_names[i]} (Pump Shut-off)')
            ax.plot(data_hammer[:,0], data_hammer[:,1+i], '--', linewidth=4, label='HAMMER', color='#5a8ce6')
            ax.plot(data_tsnet[:,0], data_tsnet[:,1+i], linewidth=3, label='TSNET', color=colors[1])
            ax.plot(data_ptsnet[:,0], data_ptsnet[:,1+i], linewidth=3, label='PTSNET', color=colors[2])
            plt.xlim(0, 20)
            plt.xlabel('Time [s]'); plt.ylabel('Head [m]')
            ax.grid()
            box = ax.get_position()
            ax.set_position([box.x0, box.y0 + box.height * 0.15, box.width, box.height * 0.9])
            ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), fancybox=True, shadow=True, ncol=5)
            plt.savefig(f'figures/pump_shutoff_{nodes_names[i]}.pdf')
    else:
        ax = [None, None, None]
        fig, axs = plt.subplots(3)
        for sim_num, simulator in enumerate(('TSNET','HAMMER','PTSNET')):
            # axs[sim_num].set_title(f'({chr(ord("a")+sim_num)}) '+simulator+'  ', loc='right', y=1.0, pad=-20)
            for i in range(len(nodes_names)):
                if simulator == 'HAMMER':
                    axs[sim_num].plot(data_hammer[:,0], data_hammer[:,1+i], marker=markers[i], markevery=100, markersize=6, linewidth=3, alpha=alphas[i], label=nodes_names[i].replace('JUNCTION-',""), color=grayscale[i], linestyle=lstyle[i])
                elif simulator == 'TSNET':
                    axs[sim_num].plot(data_tsnet[:,0], data_tsnet[:,1+i], marker=markers[i], markevery=100, markersize=6, linewidth=3, alpha=alphas[i], label=nodes_names[i].replace('JUNCTION-',""), color=grayscale[i], linestyle=lstyle[i])
                elif simulator == 'PTSNET':
                    axs[sim_num].plot(data_ptsnet[:,0], data_ptsnet[:,1+i], marker=markers[i], markevery=100, markersize=6, linewidth=3, alpha=alphas[i], label=nodes_names[i].replace('JUNCTION-',""), color=grayscale[i], linestyle=lstyle[i])
                plt.xlabel('Time [s]', fontsize=20); axs[sim_num].set_ylabel('Head [m]', fontsize=20)
                axs[sim_num].set_xlim(0, 5)
                axs[sim_num].set_ylim(220, 300)
                # axs[sim_num].grid()
        for ax in fig.get_axes():
            ax.label_outer()
        fig.set_size_inches(9, 12)
        # plt.subplots_adjust(hspace=0)
        plt.legend(loc='upper center', bbox_to_anchor=(0.47, -0.2), fancybox=True, shadow=True, ncol=5)
        plt.savefig(f'figures/pump.pdf')
elif sys.argv[1] == 'ptsnet':
    sim_time_0 = time()
sim = PTSNETSimulation(
    inpfile = inpfile,
    settings = {
        'duration': 20,
        'time_step': global_dt,
        'period' : 0,
        'default_wave_speed' : global_wave_speed})
    sim.define_pump_operation('PUMP-172', initial_setting=1, final_setting=0, start_time=0, end_time=1)
    sim.add_surge_protection('JUNCTION-34', 'closed', 0.1, 1, 0.2)
    # sim.add_surge_protection('JUNCTION-34', 'open', 0.1)
    print(sim.time_step)
    sim.run()
    sim_time_1 = time()
    rslts = [sim['node'].head[node] for node in nodes_names]
    np.savetxt(
        'results/pump_ptsnet.txt',
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
    tm.pump_shut_off('PUMP-172', [1,0,0,1])
    tm = tsnet.simulation.Initializer(tm,t0)
    tm = tsnet.simulation.MOCSimulator(tm)
    sim_time_1 = time()
    tt = tm.simulation_timestamps
    rslts = []
    for node in nodes_names:
        nd = tm.get_node(node)
        rslts.append(nd.head)
    np.savetxt(
        'results/pump_tsnet.txt',
        list(zip(tt, *rslts)),
        delimiter=',',
        header='Time,'+','.join(nodes_names),
        comments='')
    print(f'Duration: {sim_time_1-sim_time_0} s')


