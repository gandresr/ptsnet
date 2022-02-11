import matplotlib.pyplot as plt
import networkx.drawing.nx_pylab as nxp
import numpy as np
import bisect

from matplotlib.lines import Line2D
from ptsnet.utils.analytics import compute_wave_speed_error

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
