import importlib.util

pmoc_path = "/home/watsup/Documents/Github/hammer-net/partitioning/pmoc.py"
spec = importlib.util.spec_from_file_location("pmoc", pmoc_path)
pmoc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pmoc)
Mesh = pmoc.Mesh

test_mesh = Mesh('models/LoopedNet.inp', 0.01)

results_test_1 = {
    '0': 100, '1': 100, '2': 100, '3': 100, '4': 100,
    '5': 100, '6': 100, '7': 100, '8': 100
    }

result_test_2 = {
    '0': 100, '1': 100, '2': 100, '3': 100, '4': 120,
    '5': 100, '6': 100, '7': 123.1231, '8': 100
    }


result_test_3 = {
    '0': 1500, '1': 1500, '2': 1500, '3': 1500, '4': 120,
    '5': 1500, '6': 1500, '7': 123.1231, '8': 1500
    }

# Test 1
test_mesh._define_wave_speeds(100)
assert test_mesh.wave_speeds == results_test_1

# Test 2
test_mesh._define_wave_speeds(wave_speed_file='wavespeeds.csv')
assert test_mesh.wave_speeds == result_test_2

# Test 3
test_mesh._define_wave_speeds(1500, wave_speed_file='wavespeeds.csv')
assert test_mesh.wave_speeds == result_test_3
