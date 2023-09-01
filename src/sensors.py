from regressors import PModel, CSModel
import numpy as np



class Sensor():
    """
    Abstract base class for sensors.
    """
    def __init__(
            self, 
            noise_dev :float, 
            offset_function :callable, 
            failure_chance :float, 
            value_range :np.ndarray[float],
            relative_sites :np.ndarray[float]
        ) -> None:
        """
        Args:
            noise_dev (float): standard deviation of normally distributed noise.
            offset_function (callable): systematic error addition function.
            failure_chance (float): chance of sensor failing.
            value_range (np.ndarray[float]): 2 by m array of lower and upper bounds of values of dimension m.
            relative_sites (np.ndarray[float]): n by d array of n relative positions of dimension d=1 or d=2 to measure sensor values at.
        """
        self._noise_dev = noise_dev
        self._offset_function = offset_function
        self._failure_chance = failure_chance
        self._range = value_range

        self._relative_sites = relative_sites
        self._value_dim = len(value_range[0])

    
    def get_failure_chance(self) -> float:
        """
        Returns:
            float: chance of the sensor failing.
        """
        return self._failure_chance


    def get_input_sites(self, actual_pos :np.ndarray[float]) -> np.ndarray[float]:
        """
        Args:
            actual_pos (np.ndarray[float]): 1 by d array of the actual 1 or 2D sensor position.

        Returns:
            np.ndarray[float]: n by d array of the sensor sampling positions.
        """
        return self._relative_sites + actual_pos*np.ones(self._relative_sites.shape)

    
    def get_output_values(self, site_values :np.ndarray[float], actual_pos :np.ndarray[float]) -> tuple[np.ndarray]:
        """
        Args:
            site_values (np.ndarray[float]): n by m array of the true field values at the sampling sites.
            actual_pos (np.ndarray[float]): 1 by d array of the actual sensor position.

        Returns:
            tuple[np.ndarray]: first element is the sensor output, second element is the sensor position(s) from which this output is taken.
        """
        mean_value = np.mean(site_values, axis=0).reshape(-1, self._value_dim)
        mean_value = self._squash_to_range(mean_value)
        noise_array = np.random.normal(0, self._noise_dev, size=mean_value.size)

        out_pos = np.expand_dims(actual_pos, axis=0)
        out_value = mean_value + noise_array + self._offset_function(mean_value)
        return (out_value, out_pos)


    def _squash_to_range(self, array) -> np.ndarray[float]:
        """
        Given an array, clip all the values outside the range into the range.

        Args:
            array (_type_): n by m array of n true sensor values of dimension m.

        Returns:
            np.ndarray[float]: clipped array.
        """
        for i, value in enumerate(array):
            if np.any(value < self._range[0]):
                array[i] = self._range[0]
            elif np.any(array > self._range[1]):
                array[i] = self._range[1]
        return array



class PointSensor(Sensor):
    """
    Point sensor samples 1 point only.
    Used for 2D fields.
    """
    def __init__(
            self, 
            noise_dev :float, 
            offset_function :callable, 
            failure_chance :float, 
            value_range :np.ndarray[float],
            field_dim :int
        ) -> None:
        """
        Args:
            noise_dev (float): standard deviation of normally distributed noise.
            offset_function (callable): systematic error addition function.
            failure_chance (float): chance of sensor failing.
            value_range (np.ndarray[float]): 2 by m array of lower and upper bounds of values of dimension m.
        """
        if field_dim == 2:
            measurement_sites = np.array([[0, 0]])
        elif field_dim == 1:
            measurement_sites = np.array([[0]])
        else:
            raise Exception('Can only have 1D or 2D field dimensions.')
        super().__init__(noise_dev, offset_function, failure_chance, value_range, measurement_sites)



class RoundSensor(Sensor):
    """
    Round sensor samples 5 points in a cross shape.
    Used for 2D fields.
    """
    def __init__(
            self, 
            noise_dev :float, 
            offset_function :callable, 
            failure_chance :float, 
            value_range :np.ndarray[float],
            radius :float,
            field_dim :int
        ) -> None:
        """
        Args:
            noise_dev (float): standard deviation of normally distributed noise.
            offset_function (callable): systematic error addition function.
            failure_chance (float): chance of sensor failing.
            value_range (np.ndarray[float]): 2 by m array of lower and upper bounds of values of dimension m.
            radius (float): radius of cross (radius of sensor in real life).
        """
        if field_dim == 2:
            measurement_sites = np.array([[0, 0], [0, radius], [0, -radius], [-radius, 0], [radius, 0]])
        elif field_dim == 1:
            measurement_sites = np.array([[0], [0], [0], [-radius], [radius]])
        else:
            raise Exception('Can only have 1D or 2D field dimensions.')
        super().__init__(noise_dev, offset_function, failure_chance, value_range, measurement_sites)



class MultiSensor(Sensor):
    """
    Multi-sensor samples many regions in a grid described.
    It then returns many values - 1 for each point in the grid.
    Designed to act as a parent class for things like a DIC or an IR camera.
    It doesn't have a meaningful position - the input sites are the same regardless of the actual_pos.
    Used for 1D or 2D fields.
    """
    def __init__(
            self, 
            noise_dev :float, 
            offset_function :callable, 
            failure_chance :float, 
            value_range :np.ndarray[float],
            grid :np.ndarray[float]
        ) -> None:
        """
        Args:
            noise_dev (float): standard deviation of normally distributed noise.
            offset_function (callable): systematic error addition function.
            failure_chance (float): chance of sensor failing.
            value_range (np.ndarray[float]): 2 by m array of lower and upper bounds of values of dimension m.
        """
        super().__init__(noise_dev, offset_function, failure_chance, value_range, grid)

    
    def get_input_sites(self, actual_pos :np.ndarray[float]) -> np.ndarray[float]:
        """
        Args:
            actual_pos (np.ndarray[float]): 1 by d array of the actual 1 or 2D sensor position.

        Returns:
            np.ndarray[float]: n by d array of the sensor sampling positions.
        """
        return self._relative_sites

    
    def get_output_values(self, site_values :np.ndarray[float], actual_pos :np.ndarray[float]) -> tuple[np.ndarray]:
        """
        Args:
            site_values (np.ndarray[float]): n by m array of the true field values at the sampling sites.
            actual_pos (np.ndarray[float]): 1 by d array of the actual sensor position.

        Returns:
            tuple[np.ndarray]: first element is the sensor output, second element is the sensor position(s) from which this output is taken.
        """
        squashed_values = self._squash_to_range(site_values)
        noise_array = np.random.normal(0, self._noise_dev, size=squashed_values.size)

        out_value = squashed_values + noise_array + self._offset_function(squashed_values)
        return (out_value, self._relative_sites)




class Thermocouple(RoundSensor):
    """
    Round sensor with a linearisation error.
    Used for 2D fields.
    """
    def __init__(
            self, 
            temps :np.ndarray[float], 
            voltages :np.ndarray[float], 
            field_dim :int,
            noise_dev=0.6, 
            failure_chance=0.4, 
            radius=0.00075
        ) -> None:
        """
        Args:
            temps (np.ndarray[float]): array of n temperatures to interpolate through.
            voltages (np.ndarray[float]): array of n voltages to interpolate through.
            noise_dev (float, optional): standard deviation of noise. Defaults to 0.6.
            failure_chance (float, optional): chance of failure. Defaults to 0.4.
            radius (float, optional): radius of thermocouple. Defaults to 0.00075.
        """
        self._regressor = PModel(1, degree=1)
        self._regressor.fit(voltages, temps)
        self._interpolator = CSModel(1)
        self._interpolator.fit(temps, voltages)

        value_range = np.array([[min(temps)], [max(temps)]])
        super().__init__(noise_dev, self.non_linear_error, failure_chance, value_range, radius, field_dim)

    
    def non_linear_error(self, temp :np.ndarray[float]) -> np.ndarray[float]:
        voltage = self._interpolator.predict(temp)
        new_temp = self._regressor.predict(voltage)
        return new_temp - temp