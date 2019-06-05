import importlib.util
import os
import unittest

pmoc_path = "/home/watsup/Documents/Github/hammer-net/parallel_moc/pmoc.py"
spec = importlib.util.spec_from_file_location("pmoc", pmoc_path)
pmoc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pmoc)
Mesh = pmoc.Mesh


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

test_mesh = None

class MeshTests(unittest.TestCase):
    def tests(self):
        test_mesh = Mesh(
            os.getcwd() + os.sep + 'models/LoopedNet.inp',
            dt = 0.01,
            default_wave_speed = 100)

        # Test 1
        # Wave speeds are defined based on default wave speed value
        self.assertEqual(test_mesh.wave_speeds, results_test_1)

        # Test 2
        # Wave speeds are redefined based on file and default wave speed value
        test_mesh._define_wave_speeds(
            default_wave_speed = 100, wave_speed_file='wavespeeds.csv')
        self.assertEqual(test_mesh.wave_speeds, result_test_2)

        # Test 3
        # Incomplete file, no default wave speed
        with self.assertRaises(Exception) as context:
            test_mesh._define_wave_speeds(wave_speed_file='wavespeeds.csv')
        self.assertTrue("""
            The file does not specify wave speed values for all the pipes,
            it is necessary to define a default wave speed value""" in str(context.exception))
        self.assertEqual(test_mesh.wave_speeds, {})

if __name__ == '__main__':
    unittest.main()