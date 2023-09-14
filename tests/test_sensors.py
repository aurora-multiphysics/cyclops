import sys
import os

# Sort the paths out to run from this file
parent_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
src_path = os.path.join(os.path.sep,parent_path, 'src')
sys.path.append(src_path)

from sensors import Sensor
import numpy as np
import unittest




class Test(unittest.TestCase):
    def test_sensor_scalar(self):
        def f(x): return np.zeros(x.shape)
        sensor = Sensor(
            0.0,
            f,
            0.1,
            np.array([[-500], [500]]),
            np.array([[0, 0], [1, 1], [-1, -1]])
        )
        self.assertEqual(sensor.get_failure_chance(), 0.1)

        sensor_sites = sensor.get_input_sites(np.array([5, 5]))
        true_sites = np.array([[5, 5], [6, 6], [4, 4]])
        for i, pos in enumerate(sensor_sites):
            self.assertEqual(pos[0], true_sites[i, 0])
            self.assertEqual(pos[1], true_sites[i, 1])
        
        true_values = [[1], [2], [3]]
        sensor_values, sensor_pos = sensor.get_output_values(true_values, np.array([5, 5]))
        self.assertEqual(sensor_values[0, 0], 2)
        self.assertEqual(sensor_pos[0, 0], 5)
        self.assertEqual(sensor_pos[0, 1], 5)


    def test_sensor_vector(self):
        def f(x): return np.zeros(x.shape)
        sensor = Sensor(
            0.0,
            f,
            0.1,
            np.array([[-500, -500], [500, 500]]),
            np.array([[0, 0], [1, 1], [-1, -1]])
        )
        self.assertEqual(sensor.get_failure_chance(), 0.1)

        sensor_sites = sensor.get_input_sites(np.array([5, 5]))
        true_sites = np.array([[5, 5], [6, 6], [4, 4]])
        for i, pos in enumerate(sensor_sites):
            self.assertEqual(pos[0], true_sites[i, 0])
            self.assertEqual(pos[1], true_sites[i, 1])
        
        true_values = [[1, 1], [2, 1], [3, 1]]
        sensor_values, sensor_pos = sensor.get_output_values(true_values, np.array([5, 5]))
        self.assertEqual(sensor_values[0, 0], 2)
        self.assertEqual(sensor_values[0, 1], 1)
        self.assertEqual(sensor_pos[0, 0], 5)
        self.assertEqual(sensor_pos[0, 1], 5)


if __name__ == "__main__":
    unittest.main()