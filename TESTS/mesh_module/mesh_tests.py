import importlib.util
import os
import unittest
from collections import defaultdict as dd

pmoc_path = "/home/watsup/Documents/Github/hammer-net/parallel_moc/pmoc.py"
spec = importlib.util.spec_from_file_location("pmoc", pmoc_path)
pmoc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pmoc)
Mesh = pmoc.Mesh

class MeshTests(unittest.TestCase):
    def test_mesh_creation(self):

        # Expected Results

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

    def test_segment_definition(self):
        test_mesh = Mesh(
            os.getcwd() + os.sep + 'models/LoopedNet.inp',
            dt = 0.01,
            default_wave_speed = 1500)

        # Test 1
        test_mesh._define_segments(0.1)
        self.assertEqual(test_mesh.time_step, 0.1)

        results_test_2 = {
            '0': 1525.0, '1': 1827.9999999999995, '2': 1525.0, '3': 2284.9999999999995,
            '4': 1829.9999999999998, '5': 1677.4999999999998, '6': 1525.0,
            '7': 2284.9999999999995, '8': 1626.6666666666667
            }

        # Test 2
        self.assertEqual(test_mesh.wave_speeds, results_test_2)

        results_test_3 = {
            '0': 4, '1': 5, '2': 4, '3': 2,
            '4': 3, '5': 4, '6': 4, '7': 2, '8': 3
            }

        # Test 3
        self.assertEqual(results_test_3, test_mesh.segments)

    def test_ids_definition(self):
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

        # Test 1
        # Checks that the the number of interior and boundary
        #   points in each pipe is correct based on the number
        #   of segments of the pipe
        self.assertEqual(real_segments, test_mesh.segments)

    def test_partitioning(self):
        test_mesh = Mesh(
            os.getcwd() + os.sep + 'models/LoopedNet.inp',
            dt = 0.01,
            default_wave_speed = 10000)
        test_mesh._define_partitions(4)

if __name__ == '__main__':
    unittest.main()