"""
Tests for the cyclops SensorSuite class.

(c) Copyright UKAEA 2023.
"""
import sys
import os

# Sort the paths out to run from this file
parent_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
src_path = os.path.join(os.path.sep, parent_path, "src")
sys.path.append(src_path)

from regressors import RBFModel, LModel, GPModel, CSModel, CTModel, PModel
from sensor_group import SensorSuite, SymmetryManager
from fields import ScalarField, VectorField
from sensors import Sensor
import numpy as np
import unittest


# Key
# g means grid test (2D field plane)
# l means line test (1D field line)


class TestSensorSuite(unittest.TestCase):
    def test_sensor_suite1(self):
        def f(x):
            return np.zeros(x.shape)

        sensor = Sensor(
            0,
            f,
            0.1,
            np.array([[-50], [50]]),
            np.array([[0, 0], [-1, -1], [1, 1]]),
        )

        sensor_sequence = np.array([sensor] * 5)
        sensor_suite = SensorSuite(
            ScalarField(RBFModel, np.array([[-15, -15], [15, 15]])),
            sensor_sequence,
        )
        sensor_suite.set_active_sensors(
            np.array([True, True, False, True, True])
        )
        sensor_suite.set_sensor_pos(
            np.array([[0, 0], [1, 10], [2, -2], [3, -3], [4, -4]])
        )
        sensor_sites = sensor_suite.get_sensor_sites()

        site_values = np.sum(sensor_sites, axis=1).reshape(-1, 1)
        sensor_suite.fit_sensor_model(site_values)

        test_pos = np.array([[0, 0], [1, 10], [2, -2], [3, -3], [4, -4]])
        values = sensor_suite.predict_data(test_pos)
        print(values)


if __name__ == "__main__":
    unittest.main()
