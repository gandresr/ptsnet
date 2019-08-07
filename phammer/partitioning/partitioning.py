import subprocess
import numpy as np

from os import sep
from os.path import isdir
from pkg_resources import resource_filename

def define_partitions(graph_file, k):
    """Defines network partitioning using parHIP (external lib)

    Arguments:
        k {integer} -- desired number of partitions
    """

    RESOURCE_PATH = resource_filename(__name__, 'resources')

    if not isdir(RESOURCE_PATH + sep + 'partitions'):
        subprocess.call(['mkdir', RESOURCE_PATH + sep + 'partitions'])

    script = RESOURCE_PATH + sep + 'kaffpa'
    subprocess.call([
        script, graph_file + '.graph',
        '--k=' + str(k),
        '--preconfiguration=strong',
        '--output_filename=' + RESOURCE_PATH + sep + sep.join(('partitions', 'p%d.graph' % k))],
        stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)

    script = RESOURCE_PATH + sep + 'partition_to_vertex_separator'
    subprocess.call([
        script, graph_file + '.graph',
        '--k=' + str(k),
        '--input_partition=' + RESOURCE_PATH + sep + sep.join(('partitions', 'p%d.graph' % k)),
        '--output_filename=' + RESOURCE_PATH + sep + sep.join(('partitions', 's%d.graph' % k))],
        stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)

    partitions = np.loadtxt(RESOURCE_PATH + sep + sep.join(('partitions', 'p%d.graph' % k)), dtype=np.int)
    is_ghost = np.loadtxt(RESOURCE_PATH + sep + sep.join(('partitions', 's%d.graph' % k)), dtype=np.int)
    is_ghost = is_ghost == k

    return partitions, is_ghost