"""
Tests for the cyclops SensorSuite class.

(c) Copyright UKAEA 2023.
"""
import numpy as np
import unittest

from cyclops.regressors import RBFModel, LModel, GPModel, CSModel, CTModel, PModel
from cyclops.sensor_suite import SensorSuite, SymmetryManager
from cyclops.fields import ScalarField, VectorField
from cyclops.sensors import Sensor


class TestSensorSuite(unittest.TestCase):
    """Tests for SensorSuite."""

    def test_sensor_suite(self):
        """Test that sensor suite core functionality works as expected."""
        sensor = Sensor(
            0,
            lambda x: np.zeros(x.shape),
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
        sensor_suite.predict_data(test_pos)


if __name__ == "__main__":
    unittest.main()
