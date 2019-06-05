import importlib.util
import os
from collections import defaultdict as dd

pmoc_path = "/home/watsup/Documents/Github/hammer-net/parallel_moc/pmoc.py"
spec = importlib.util.spec_from_file_location("pmoc", pmoc_path)
pmoc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pmoc)
Mesh = pmoc.Mesh

test_mesh = Mesh(
    os.getcwd() + os.sep + 'models/LoopedNet.inp',
    dt = 0.01,
    default_wave_speed = 10000)
test_mesh._define_segments(0.5)

real_segments = dd(int)
for node in test_mesh.node_ids:
    if '.' in node:
        labels = node.split('.')
        start = labels[0]
        k = labels[1]
        end = labels[2]
        pipe = labels[3]
        real_segments[pipe] += 1
for pipe in real_segments:
    real_segments[pipe] -= 1

assert real_segments == test_mesh.segments
print("OK - Test 1")