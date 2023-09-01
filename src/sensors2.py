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
            measurement_sites :np.ndarray[float]
        ) -> None:
        """
        Args:
            noise_dev (float): standard deviation of normally distributed noise.
            offset_function (callable): systematic error addition function.
            failure_chance (float): chance of sensor failing.
            value_range (np.ndarray[float]): 2 by m array of lower and upper bounds of values of dimension m.
            measurement_sites (np.ndarray[float]): n by 1 or 2 array of n 1/2D relative positions to measure sensor values at.
        """
        self._noise_dev = noise_dev
        self._offset_function = offset_function
        self._failure_chance = failure_chance
        self._range = value_range

        self._measurement_sites = measurement_sites
        self._true_site_values = np.zeros(len(measurement_sites)).reshape(-1, 1)
        num_dim = len(value_range[0])
        self._num_dim = num_dim

    
    def get_failure_chance(self) -> float:
        """
        Returns:
            float: chance of the sensor failing.
        """
        return self._failure_chance


    def get_sites(self) -> np.ndarray[float]:
        """
        Returns:
            np.ndarray[float]: n by 1 or 2 array of n 1/2D measurement positions.
        """
        return self._measurement_sites

    
    def set_values(self, values: np.ndarray[float]) -> None:
        """
        Args:
            values (np.ndarray[float]): n by m array of n true values of dimension m.
        """
        self._true_site_values = values

    
    def get_values(self) -> np.ndarray[float]:
        """
        Returns:
            np.ndarray[float]: 1 by m array of the measured sensor value of dimension m. 
        """
        out_value = np.mean(self._true_site_values, axis=0).reshape(-1, self._num_dim)
        out_value = self._squash_to_range(out_value)
        noise_array = np.random.normal(0, self._noise_dev, size=out_value.size)
        return out_value + noise_array + self._offset_function(out_value)


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

    
    def get_values(self):
        noise_array = np.random.normal(0, self._noise_dev, size=self._true_site_values.shape)
        return self._true_site_values + noise_array + self._offset_function(self._true_site_values)




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

        value_range = np.array([min(temps), max(temps)])
        super().__init__(noise_dev, self.non_linear_error, failure_chance, value_range, radius, field_dim)

    
    def non_linear_error(self, temp :np.ndarray[float]) -> np.ndarray[float]:
        """
        Calculates the voltage produced by the thermocouple.
        Then the linearised temperature measurement from the voltage.
        Then the systematic error.

        Args:
            temp (float): temperature value.

        Returns:
            float: systematic error at that temperature.
        """
        voltage = self._interpolator.predict(np.array(temp))
        new_temp = self._regressor.predict(voltage)
        return new_temp - temp