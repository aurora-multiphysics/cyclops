"""
SensorSuite class for cyclops.

Handles the various sensors employed by an experiment.

(c) Copyright UKAEA 2023.
"""
import numpy as np

from cyclops.sensors import Sensor, PointSensor, RoundSensor
from cyclops.fields import Field, ScalarField, VectorField


class SensorSuite:
    """Class for a sensor suite.
    Holds the sensor postions & types and allows for a field to be predicted
    from the sensor data.
    """

    def __init__(self, true_field: Field, sensors: list, sensor_pos: list,
                field_points: np.ndarray, field_vector_vals: np.ndarray) -> None:
        """Initialise class instance.

        Args:
            true_field (Field): the simulated field which acts as the ground
                truth against which to compare the predicted field.
            comparison_pos (np.ndarray[float]): the positions used to
                compare the true field to the predicted field.
            sensors (list): a list of sensor types to populate the suite with,
            must be in the same order as the intended position appears in
            'sensor_pos'
            sensor_pos (list): a list of 3D positions, each representing the
            centre point of the corresponding sensor in 'sensors'
        """
        self.__true_field = true_field
        self.__sensors = sensors
        self.__num_sensors = len(self.__sensors)
        #self.__active_sensors = np.full(self.__num_sensors, True)
        #self.__sensor_pos = sensor_pos
        self.__field_points = field_points
        self.__field_vector_vals = field_vector_vals


    # def set_active_sensors(self, active_sensors: np.ndarray[bool]):
    #     """Set which sensors are active.

    #     Args:
    #         active_sensors (np.ndarray[bool]): array of booleans to show which
    #             sensors are off or on.
    #     """
    #     self.__active_sensors = active_sensors

    # def set_sensor_pos(self, sensor_pos: np.ndarray[float]):
    #     """Set the positions of the sensors.

    #     Args:
    #         sensor_pos (np.ndarray[float]): n by d array of n positions of d
    #             dimensions.
    #     """
    #     self.__sensor_pos = sensor_pos

    def get_sensor_outputs(self) -> np.ndarray[float]:
        """Return the positions from which the sensors sample from.

        Returns:
            np.ndarray[float]: n by d array of n positions of d dimensions to
                sample from.
        """
        #Need the field values at sensor read in points
        absolute_sites = []
        all_observs = []
        
        for i in range(len(self.__sensors)):

            current_snr = self.__sensors[i]

            # Get the sampling sites for this sensor
            sensor_read_in = current_snr.get_measurement_sites()
            # Fit the true field to the sensor
            if isinstance(self.__true_field, Field):
                field_fit = self.__true_field
                real_vals = field_fit.predict_values(sensor_read_in)

            # Get the sampled values and adjusted sampling sites
            sensor_obvs, sensor_pos = current_snr.get_output_values( 
                    true_site_values=real_vals,
                    actual_pos=sensor_read_in)
            
            absolute_sites.append(sensor_read_in)
            all_observs.append(sensor_obvs)

        absolute_sites = np.array(absolute_sites, dtype=object)
        all_observs = np.array(all_observs, dtype=object)

        absolute_sites.flatten
        all_observs.flatten

        return (all_observs, absolute_sites)


    def fit_sensor_model(self, known_pos: np.ndarray[float],
                         known_values: np.ndarray[float]):
        """Fit the model based on the sensor data.

        Args:
            site_values (np.ndarray[float]): n by m array of the n values of
                dimension m at the sites specified.
        """

        self.__true_field.fit_model(known_pos, known_values)

    def predict_data(self, comparison_pos: np.ndarray[float]) -> np.ndarray[float]:
        """Predict values of the field at the points specified.

        Args:
            field_pos (np.ndarray[float]): n by d array of n positions of
                dimension d.

        Returns:
            np.ndarray[float]: n by m array of n values of dimension m.
        """
        target_field = self.__true_field
        pos_3D = self.__field_points
        field_vector_vals = self.__field_vector_vals
        target_field.fit_model(pos_3D, field_vector_vals)
        target_field.predict_values(comparison_pos)

        return target_field.predict_values(comparison_pos)
#ToDo update and incorporate this aspect - may need to be in a different file/class
    # def calc_keys(self, num_repetitions: int) -> np.ndarray[bool]:
    #     """Calculate potential sensor arrays.

    #     Calculate a number of potential arrays for the active sensors based
    #     off the chances that the sensors fail.

    #     Args:
    #         num_repetitions (int): number of keys needed.

    #     Returns:
    #         np.ndarray[bool]: n by s array of n keys of dimension s where s is
    #             the number of sensors.
    #     """
    #     keys = np.full((num_repetitions, self.__num_sensors), True)
    #     for i, key in enumerate(keys):
    #         for j in range(len(key)):
    #             num = np.random.rand()
    #             if num < self.__sensors[j].get_failure_chance():
    #                 keys[i, j] = False
    #     return keys

    def get_num_sensors(self):
        """Return the number of sensors."""
        return self.__sensors.size
