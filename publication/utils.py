import matplotlib.pyplot as plt
import numpy as np

def plot_comparison_results(nodes_names, data_ptsnet, data_tsnet, data_hammer, bw=False):
    if bw:
        colors = ['#000000', '#333333', '#666666', '#999999', '#cccccc'] # bw
    else:
        colors = ['#007aff', '#4cd964', '#ff9500', '#ff3b30', '#5856d6']

    markers = ['*', 'x', '.', '', 'o']
    lstyle = ['-','--', '-', ':', '-']
    markers = ['', '', '', '', '']
    alphas = [0.7, 1, 1, 1, 1]

    fig, axs = plt.subplots(3)
    for sim_num, simulator in enumerate(('TSNET','HAMMER','PTSNET')):
        axs[sim_num].set_title(f'({chr(ord("a")+sim_num)}) '+simulator+'  ', loc='right', y=1.0, pad=-20)
        for i, node_name in enumerate(nodes_names):
            if simulator == 'HAMMER':
                axs[sim_num].plot(
                    data_hammer['Time'],
                    data_hammer[node_name],
                    marker=markers[i],
                    markevery=100,
                    markersize=6,
                    linewidth=3,
                    alpha=alphas[i],
                    label=node_name.replace('JUNCTION-',""),
                    color=colors[i], linestyle=lstyle[i])
            elif simulator == 'TSNET':
                axs[sim_num].plot(
                    data_tsnet['Time'],
                    data_tsnet[node_name],
                    marker=markers[i],
                    markevery=100,
                    markersize=6,
                    linewidth=3,
                    alpha=alphas[i],
                    label=nodes_names[i].replace('JUNCTION-',""),
                    color=colors[i], linestyle=lstyle[i])
            elif simulator == 'PTSNET':
                axs[sim_num].plot(
                    data_ptsnet['Time'],
                    data_ptsnet[node_name],
                    marker=markers[i],
                    markevery=100,
                    markersize=6,
                    linewidth=3,
                    alpha=alphas[i],
                    label=nodes_names[i].replace('JUNCTION-',""),
                    color=colors[i], linestyle=lstyle[i])
            plt.xlabel('Time [s]', fontsize=20); axs[sim_num].set_ylabel('Head [m]', fontsize=20)
            axs[sim_num].set_xlim(0, data_ptsnet['Time'].iloc[-1])
    for ax in fig.get_axes():
        ax.label_outer()
    fig.set_size_inches(9, 12)
    # plt.subplots_adjust(hspace=0)
    plt.legend(loc='upper center', bbox_to_anchor=(0.47, -0.2), fancybox=True, shadow=True, ncol=5)
    plt.show()