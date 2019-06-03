import importlib.util

pmoc_path = "/home/watsup/Documents/Github/hammer-net/partitioning/pmoc.py"
spec = importlib.util.spec_from_file_location("pmoc", pmoc_path)
pmoc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pmoc)
Mesh = pmoc.Mesh

test_mesh = Mesh('models/LoopedNet.inp', 0.01, default_wave_speed = 1500)


test_mesh._define_segments(0.1)
assert test_mesh.time_step == 0.1

results_test_2 = {
    '0': 1525.0, '1': 1827.9999999999995, '2': 1525.0, '3': 2284.9999999999995,
    '4': 1829.9999999999998, '5': 1677.4999999999998, '6': 1525.0,
    '7': 2284.9999999999995, '8': 1626.6666666666667
    }

assert test_mesh.wave_speeds == results_test_2

results_test_3 = {
    '0': 4, '1': 5, '2': 4, '3': 2,
    '4': 3, '5': 4, '6': 4, '7': 2, '8': 3
    }

assert results_test_3 == test_mesh.segments