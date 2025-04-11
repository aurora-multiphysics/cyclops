"""
Sensor classes for cyclops.

Handle sensor properties and can emulate exact or noisy sensor readings.

(c) Copyright UKAEA 2023.
"""
import numpy as np

from cyclops.regressors import PModel, CSModel
from cyclops.fields import Field, ScalarField, VectorField
from scipy.spatial.transform import Rotation as R

#ToDo MultiSensor and ThermoSensor classes have not been updated

class Sensor:
    """Abstract base class for sensors."""

    def __init__(
        self,
        offset_function: callable,
        field_dim: float,
        field: Field,
        centre_point: np.ndarray,
        noise_dev=0,        
        failure_chance=0
    ) -> None:
        """Initialise class instance.

        Args:
            offset_function (callable): systematic error addition function.
            field_dim (int): dimensionality of the field.
            field (Field): the field to be measured.
            sensor_types (list): list of sensor types.
            centre_points (np.ndarray): sensor center points.
            noise_dev (float): standard deviation of normally distributed noise.
            failure_chance (float): chance of sensor failing.
        """
        self._noise_dev = noise_dev
        self._offset_function = offset_function
        self._failure_chance = failure_chance
        self._field_dim = field_dim
        self._field = field
        self._centre_point = centre_point
        # Set for default single-point sensor
        self._relative_sites = np.zeros((1, field_dim))
        # Determined based on sensor type later
        self._measurement_sites = None  

    def move_sensor(self, new_centre_point: np.ndarray) -> None:
        """
        Move sensor to a new position and update measurement sites accordingly
        
        Args:

        Returns:
        """
        self._centre_point = np.array(new_centre_point)

        # Update measurement sites (should be overridden in subclasses if needed)
        if self._measurement_sites is not None:
            shift = new_centre_point - self._centre_point
            self._measurement_sites += shift

    def get_centre_point(self) -> np.ndarray:
        """Returns the current centre point of the sensor."""
        return self._centre_point

    def get_failure_chance(self) -> float:
        """Return the chance of the sensor failing."""
        #ToDo improve how failure chance is defined, varying with temp etc
        return self._failure_chance

    def get_input_sites(self) -> np.ndarray[float]:
        """Get positions at which sensor takes readings. Takes the centre
        position of the sensor as 'actual_pos' and calculates where each
        sampling point will be.

        Args:
            actual_pos (np.ndarray[float]): 1 by d array of the actual sensor
            position.

        Returns:np.zeros((1, field_dim)) 
            np.ndarray[float]: n by d array of the sensor sampling positions.
        """
        #ToDo is this correct? Unsure
        sample_sites = self._relative_sites + self._centre_point * np.ones(
            self._relative_sites.shape)
        sample_sites = sample_sites.flatten()
        return sample_sites

    def get_output_values(
        self, true_site_values: np.ndarray, actual_pos: np.ndarray
    ) -> tuple[np.ndarray]:
        """Get sensor reading value.

        Args:
            true_site_values (np.ndarray[float]): n by m array of the true field
                values at the sampling sites.
            actual_pos (np.ndarray[float]): 1 by d array of the actual sensor
                position.

        Returns:
            tuple[np.ndarray]: first element is the sensor output, second
                element is the sensor position(s) from which this output is
                taken.
        """

        if isinstance(self._field, ScalarField):
        # For ScalarField, we use the predict_values method directly
            #regressor = self._field.fit_model()
            site_values = self._field.predict_values(actual_pos)

        elif isinstance(self._field, VectorField):
            #regressor = self._field.fit_model()
            site_values = self._field.get_values(actual_pos) 
        
        #Ensure numpy array format
        site_values = np.array(site_values)

        #Check that site_values is not empty
        if site_values.size == 0:
            raise ValueError("Error: site_values is empty!")

        #ToDo might be a more accurate way to represent how different sensors
        # aggregate their measurememts
        mean_value = np.mean(site_values)
        mean_array= np.full_like(site_values, fill_value=mean_value)

        #ToDo might want the noise function to be a user choice
        noise_array = np.random.normal(
            0, self._noise_dev, size=site_values.shape
        )

        out_pos = np.expand_dims(actual_pos, axis=0)
        # Need to account for the noise and offset function before returning
        # values.
        out_value = (
            mean_array + noise_array + self._offset_function(mean_value)
        )

        return (out_value, out_pos)

    def get_num_input_sites(self, actual_pos: np.ndarray) -> np.ndarray:
        """Return number of sites needed to be considered for the sensor."""
        return self._relative_sites + actual_pos

    def get_measurement_sites(self) -> np.ndarray:
        """Return the sensor's measurement sites."""
        if self._measurement_sites is None:
            raise NotImplementedError("Measurement sites must be defined in a subclass.")
        return self._measurement_sites

    @staticmethod
    def rotate_to_mesh_face(measurement_sites, normal_vector):
        """Takes an input array of sensor measurement sites and a normal
        vector for a mesh face then rotates the sensor measurement
        sites to align with the face."""
        # Normalise the normal vector
        normal_vector = normal_vector/np.linalg.norm(normal_vector)

        # Want a rotation matrix that aligns z-axis of sensor with the normal
        # vector. Calculate axis of rotation (Xproduct between [0, 0, 1] and
        # norm)
        axis = np.cross([0, 0, 1], normal_vector)
        axis_norm = np.linalg.norm(axis)
        
        # If the normal is already aligned with the z-axis, no rotation is needed
        if axis_norm < 1e-6:
            return measurement_sites
        
        # Normalise the rotation axis
        axis /= axis_norm
        
        # Calculate the angle between the z-axis and the normal
        angle = np.arccos(np.dot([0, 0, 1], normal_vector))
        
        # Create 'rotation object'
        rotation = R.from_rotvec(axis * angle)
        
        # Rotate the measurement sites
        rotated_sites = rotation.apply(measurement_sites)
        
        return rotated_sites


class PointSensor(Sensor):
    """Point sensor; samples one point only."""

    def __init__(
        self,
        offset_function: callable,
        centre_point: np.ndarray,
        field_dim: int,
        field: Field,
        noise_dev=0.012,
        failure_chance=0.01,
    ) -> None:
        """Initialise class instance.

        Args:
            noise_dev (float): standard deviation of normally distributed
                noise.
            offset_function (callable): systematic error addition function.
            failure_chance (float): chance of sensor failing.
            value_range np.ndarray: 
        """
        print("centre_point is: ", centre_point)
        measurement_sites = centre_point

        super().__init__(offset_function, field_dim, field,
                         centre_point, noise_dev, failure_chance)
        
        self._measurement_sites = measurement_sites
    
    def move_sensor(self, new_centre_point: np.ndarray) -> None:
        """Move a PointSensor to a new location."""
        super().move_sensor(new_centre_point)
        self._measurement_sites = np.array(new_centre_point)


class RoundSensor(Sensor):
    """Round sensor; samples five points in a cross shape.

    Used for 2D fields.
    """

    def __init__(
        self,
        offset_function: callable,
        field_dim: int,
        field: Field,
        norm_vector: np.ndarray,
        centre_point: np.ndarray,
        radius=0.05,
        noise_dev=0.012,
        failure_chance=0.01,
    ) -> None:
        """Initialise class instance.

        Args:
            noise_dev (float): standard deviation of normally distributed
                noise.
            offset_function (callable): systematic error addition function.
            failure_chance (float): chance of sensor failing.
            value_range (np.ndarray[float]): 2 by m array of lower and upper
                bounds of values of dimension m.
            radius (float): radius of cross (radius of sensor in real life).
            field_dim (int): number of dimensions of the field (1 or 2).
        """
        if field_dim == 3:
            # Need a "default orientation" for the cross shape, similar to the
            # 2D example and then rotate it by some angle to make it fit to
            # the surface of a face.
            measurement_sites = np.array(
                [[0, 0, 0], [0, radius, 0], [0, -radius, 0], [-radius, 0, 0], 
                 [radius, 0, 0]]
            )
        elif field_dim == 2:
            measurement_sites = np.array(
                [[0, 0], [0, radius], [0, -radius], [-radius, 0], [radius, 0]]
            )
        elif field_dim == 1:
            measurement_sites = np.array([[0], [0], [0], [-radius], [radius]]
            )

        print("centre_point is: ", centre_point)
        #Shift centre to face centre
        measurement_sites = centre_point + measurement_sites
        rotated_sites = self.rotate_to_mesh_face(measurement_sites, norm_vector)

        super().__init__(offset_function, field_dim, field, centre_point,
                         noise_dev, failure_chance)
        
        self._measurement_sites = measurement_sites

    def move_sensor(self, new_centre_point: np.ndarray) -> None:
            """Move Sensor to a new location and update measurement sites."""
            shift = new_centre_point - self._centre_point
            self._measurement_sites += shift
            self._centre_point = new_centre_point

# class MultiSensor(Sensor):
#     """Multi-sensor; samples many regions in a grid.

#     It then returns many values - 1 for each point in the grid.
#     Designed to act as a parent class for things like a DIC or an IR camera.
#     It doesn't have a meaningful position - the input sites are the same
#     regardless of the actual_pos. Used for 1D or 2D fields.
#     """

#     def __init__(
#         self,
#         noise_dev: float,
#         offset_function: callable,
#         failure_chance: float,
#         value_range: np.ndarray,
#         grid: np.ndarray,
#     ) -> None:
#         """Initialise class instance.

#         Args:
#             noise_dev (float): standard deviation of normally distributed
#                 noise.
#             offset_function (callable): systematic error addition function.
#             failure_chance (float): chance of sensor failing.
#             value_range np.ndarray: 
#         """
#         super().__init__(
#             noise_dev, offset_function, failure_chance, value_range, grid
#         )

#     def get_input_sites(
#         self, actual_pos: np.ndarray
#     ) -> np.ndarray:
#         """Get positions at which sensor takes readings.

#         Args:
#             actual_pos (np.ndarray[float]): 1 by d array of the actual 1 or 2D
#                 sensor position.

#         Returns:
#             np.ndarray[float]: n by d array of the sensor sampling positions.
#         """
#         return self._relative_sites

#     def get_output_values(
#         self, site_values: np.ndarray, actual_pos: np.ndarray
#     ) -> tuple[np.ndarray]:
#         """Get sensor reading value.

#         Args:
#             site_values (np.ndarray[float]): n by m array of the true field
#                 values at the sampling sites.
#             actual_pos (np.ndarray[float]): 1 by d array of the actual sensor
#                 position.

#         Returns:
#             tuple[np.ndarray]: first element is the sensor output, second
#                 element is the sensor position(s) from which this output is
#                 taken.
#         """
#         squashed_values = self._squash_to_range(site_values)
#         noise_array = np.random.normal(
#             0, self._noise_dev, size=squashed_values.shape
#         )
#         out_value = (
#             squashed_values
#             + noise_array
#             + self._offset_function(squashed_values)
#         )
#         return (out_value, self._relative_sites)


# class Thermocouple(RoundSensor):
#     """Thermocouple sensor class.

#     Round sensor with a linearisation error. Used for 2D fields.
#     """

#     def __init__(
#         self,
#         temps: np.ndarray[float],
#         voltages: np.ndarray[float],
#         field_dim: int,
#         noise_dev=0.6,
#         failure_chance=0.4,
#         radius=0.00075,
#     ) -> None:
#         """Initialise class instance.

#         Args:
#             temps (np.ndarray[float]): array of n temperatures to interpolate
#                 through.
#             voltages (np.ndarray[float]): array of n voltages to interpolate
#                 through.
#             noise_dev (float, optional): standard deviation of noise. Defaults
#                 to 0.6.
#             failure_chance (float, optional): chance of failure. Defaults to
#                 0.4.
#             radius (float, optional): radius of thermocouple. Defaults to
#                 0.00075.
#         """
#         self._regressor = PModel(1, degree=1)
#         self._regressor.fit(voltages, temps)
#         self._interpolator = CSModel(1)
#         self._interpolator.fit(temps, voltages)

#         value_range = np.array([[min(temps)], [max(temps)]])
#         super().__init__(
#             noise_dev,
#             self.non_linear_error,
#             failure_chance,
#             value_range,
#             radius,
#             field_dim,
#         )

#     def non_linear_error(self, temp: np.ndarray[float]) -> np.ndarray[float]:
#         """Calculate the linearisation error produced.

#         Args:
#             temp (np.ndarray[float]): temperature to find the error at.

#         Returns:
#             np.ndarray[float]: error.
#         """
#         voltage = self._interpolator.predict(temp)
#         new_temp = self._regressor.predict(voltage)
#         return new_temp - temp
