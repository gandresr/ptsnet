import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as plticker
import networkx.drawing.nx_pylab as nxp
import bisect, os, pickle
import numpy as np

from matplotlib.lines import Line2D
from ptsnet.utils.analytics import compute_wave_speed_error
from ptsnet.utils.io import get_temp_folder

def plot_wave_speed_error(sim, image_path):
    errors = compute_wave_speed_error(sim)
    intervals = np.linspace(0, 100, 5)
    percentile_labels = [("%.1f%%" + " - " + "%.1f%%") % (intervals[i], intervals[i+1],)  for i in range(len(intervals)-1)]
    error_intervals = {sim.ss['pipe'].labels[i] : bisect.bisect_left(intervals, errors[i]) for i in range(len(sim.ss['pipe'].labels))}
    colors = ['#4CD964', '#FFCC00', '#FF9500', '#FF3830']
    widths = [1, 1.5, 2, 2.5]
    custom_lines = [Line2D([0], [0], color = colors[i], lw = widths[i]) for i in range(len(widths))]
    start_nodes = sim.ss['node'].labels[sim.ss['pipe'].start_node]
    end_nodes = sim.ss['node'].labels[sim.ss['pipe'].end_node]
    G_pipes_only = list(zip(start_nodes, end_nodes))
    node_coords = {node_name : sim.wn.get_node(node_name).coordinates for node_name in sim.ss['node'].labels}
    G = sim.wn.get_graph()

    pipe_colors = [colors[error_intervals[pipe_name]-1] for pipe_name in sim.ss['pipe'].labels]
    pipe_widths = [widths[error_intervals[pipe_name]-1] for pipe_name in sim.ss['pipe'].labels]
    fig, ax = plt.subplots(figsize=(15,25))
    nxp.draw_networkx_nodes(G, node_coords, node_size = 0, label = None)
    nxp.draw_networkx_edges(G, node_coords, edgelist = G_pipes_only, edge_color = pipe_colors, width = pipe_widths, arrows = False)
    ax.legend(custom_lines, percentile_labels, title = 'Relative Error', fontsize = '15', title_fontsize = '17')
    plt.axis('off')
    fig.savefig(image_path)

def plot_times_per_step(duration=20):
    export_path = os.path.join(get_temp_folder(), "exported_times.pkl")
    if not os.path.exists(export_path):
        raise FileExistsError("There's no file with time results. You need to run ptsnet.utils.analytics.compute_simulation_times_per_step first")
    with open(export_path, 'rb') as f:
        data = pickle.load(f)

    time_steps = data['time_steps']
    processors = data['processors']
    n = len(time_steps)
    p = len(processors)
    num_steps = duration/np.tile(time_steps,(p,1)).T
    init_times = data['init_times']
    interior_times = data['interior_times']*num_steps
    boundary_times = data['boundary_times']*num_steps
    comm_times = data['comm_times']*num_steps
    totals = init_times + interior_times + boundary_times + comm_times

    matplotlib.rc('ytick', labelsize=22)
    loc = plticker.MultipleLocator(base=20e3)
    fig, ax = plt.subplots(figsize=(18, 8), dpi = 80)
    ax.yaxis.set_major_locator(loc)
    X = np.arange(p, dtype=int)
    max_y = 1.2*np.max(np.max(totals))
    plt.ylim(0, max_y)
    plt.yticks(np.linspace(0, max_y, 5))
    width = 0.8
    patterns = ['','','']
    bars1_1 = init_times + interior_times
    bars2_1 = bars1_1 + boundary_times

    for i in range(n-1,-1,-1):
        p1 = ax.bar(X-(n-i)*width/n, init_times[i], hatch=patterns[i], width = width/n, color = '#000', alpha = 1)
        p2 = ax.bar(X-(n-i)*width/n, interior_times[i], bottom = init_times[i], hatch=patterns[i], width = width/n, color = '#999', alpha = 1)
        p3 = ax.bar(X-(n-i)*width/n, boundary_times[i], bottom = bars1_1[i], hatch=patterns[i], width = width/n, color = '#ccc', alpha = 1)
        p4 = ax.bar(X-(n-i)*width/n, comm_times[i], bottom = bars2_1[i], hatch=patterns[i], width = width/n, color = '#eee', alpha = 1)

        for r1, r2, r3, r4 in zip(p1, p2, p3, p4):
            h1 = r1.get_height()
            h2 = r2.get_height()
            h3 = r3.get_height()
            h4 = r4.get_height()
            # plt.text(r1.get_x()+r1.get_width()/2., h1+h2+h3+h4, f'$\\tau_{i+1}$', ha = 'center', va='bottom', fontsize=22)
            plt.text(r1.get_x()+r1.get_width()/2., h1+h2+h3+h4, '{:,} s'.format(int(h1+h2+h3+h4)), ha = 'center', va='bottom', fontsize=16)
    plt.xticks(X+width/2-1, processors, fontsize = 22)
    box = ax.get_position()
    ax.set_position([box.x0, box.y0 + box.height * 0.1,
                    box.width, box.height * 0.9])
    plt.legend(
        (p1[0], p2[0], p3[0], p4[0],),
        (
            'Initialization',
            'Interior Points',
            'Boundary Conditions',
            'Communication'
        ),
        loc = 'upper center',
        bbox_to_anchor = (0.5, -0.15),
        fancybox = True,
        shadow = True,
        ncol = 4,
        fontsize = 18)

    plt.xlabel('Number of processors', fontsize = 20)
    ax.get_yaxis().set_major_formatter(
        matplotlib.ticker.FuncFormatter(lambda x, p: format(int(x), ',')))
    plt.ylabel('Time [s]', fontsize = 20)
    plt.grid(True)
    fig.tight_layout()
    fig.subplots_adjust(top = 0.95, bottom = 0.2, left = 0.07)
    plt.savefig('times_bwsn.pdf')
    plt.show()