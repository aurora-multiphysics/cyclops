from regressors import CSModel, PModel
import numpy as np



class Sensor():
    """
    Abstract class to describe the behaviour of sensors.
    """
    def __init__(
            self, 
            noise_dev :float, 
            offset_function :callable, 
            area_1D :np.ndarray[float], 
            area_2D :np.ndarray[float], 
            failure_chance :float, 
            value_range :np.ndarray[float]
        ) -> None:
        """
        Args:
            noise_dev (float): standard deviation of normally distributed noise.
            offset_function (callable): function that describes how much systematic error to add for different ground truth values.
            area_1D (np.ndarray[float]): n by 1 array of n relative 1D position(s) to sample from in 1D.
            area_2D (np.ndarray[float]): n by 2 array of n relative 2D position(s) to sample from in 2D.
            failure_chance (float): chance of sensor failing.
            value_range (np.ndarray[float]): array of 2 values giving the range of sensor ground truth values.
        """
        self._noise_dev = noise_dev
        self._offset_function = offset_function
        self._ground_truth = None
        self._failure_chance = failure_chance
        self._area_1D = area_1D
        self._area_2D = area_2D
        self._range = value_range

    
    def set_value(self, ground_truth_array :np.ndarray[float]) -> None:
        """
        Args:
            ground_truth_array (np.ndarray[float]): ground truth value of the sensor.
        """
        self._ground_truth = np.mean(ground_truth_array)

    
    def get_failure_chance(self) -> float:
        """
        Returns:
            float: chance sensor fails.
        """
        return self._failure_chance
    

    def get_value(self) -> float:
        """
        Calculates measured value of sensor.
        Only returns a meaningful value if in correct range.
        Considers effect of noise and systematic error.

        Returns:
            _type_: value measured by the sensor.
        """
        if self._ground_truth < self._range[0]:
            return self._range[0]
        elif self._ground_truth > self._range[1]:
            return self._range[1]
        else:
            return self._ground_truth + np.random.normal()*self._noise_dev + self._offset_function(self._ground_truth)


    def get_area(self, dim :int) -> np.ndarray[float]:
        """
        Returns the relative positions the sensor samples from.

        Args:
            dim (int): dimensions of the field we are sampling from (1 or 2).

        Returns:
            np.ndarray[float]: n by d array of n relative postions with d dimensions.
        """
        if dim == 1: 
            return self._area_1D
        else: 
            return self._area_2D

    
    def get_num_values(self) -> int:
        """
        Returns:
            int: the number of relative positions the sensor samples from.
        """
        return len(self._area_1D)




class PointSensor(Sensor):
    """
    Point sensor samples 1 point only.
    """
    def __init__(
            self, 
            noise_dev :float, 
            offset_function :callable, 
            failure_chance :float, 
            value_range :np.ndarray[float]
        ) -> None:
        """
        Defines the sample regions to sample from 1 point only.

        Args:
            noise_dev (float): standard deviation of normally distributed noise.
            offset_function (callable): function that describes how much systematic error to add for different ground truth values.
            failure_chance (float): chance of sensor failing.
            value_range (np.ndarray[float]): array of 2 values giving the range of sensor ground truth values.
        """
        area_1D = np.array([[0]])
        area_2D = np.array([[0, 0]])
        super().__init__(
            noise_dev, 
            offset_function,
            area_1D,
            area_2D,
            failure_chance,
            value_range
        )



class MultiValueSensor(Sensor):
    """
    This sensor returns many values, so can be used to simulate a DIC or an IR camera.
    """
    def __init__(
            self, 
            noise_dev :float, 
            offset_function :callable, 
            failure_chance :float, 
            value_range :np.ndarray[float],
            sites_1D :np.ndarray[float],
            sites_2D :np.ndarray[float]
        ) -> None:
        """
        Args:
            noise_dev (float): standard deviation of normally distributed noise.
            offset_function (callable): function that describes how much systematic error to add for different ground truth values.
            failure_chance (float): chance of sensor failing.
            value_range (np.ndarray[float]): array of 2 values giving the range of sensor ground truth values.
            sites_1D (np.ndarray[float]): sites relative to the position that the multi-value-sensor samples from in 1D.
            sites_2D (np.ndarray[float]): sites relative to the position that the multi-value-sensor samples from in 2D.
        """
        super().__init__(
            noise_dev, 
            offset_function,
            sites_1D,
            sites_2D,
            failure_chance,
            value_range
        )




class RoundSensor(Sensor):
    """
    Round sensor considers 5 points in a cross shape and averages them.
    """
    def __init__(
            self, 
            noise_dev :float, 
            offset_function :callable, 
            failure_chance :float, 
            value_range :np.ndarray[float], 
            radius :float
        ) -> None:
        """
        Defines the sample regions to sample from 5 points in a cross shape.

        Args:
            noise_dev (float): standard deviation of normally distributed noise.
            offset_function (callable): function that describes how much systematic error to add for different ground truth values.
            failure_chance (float): chance of sensor failing.
            value_range (np.ndarray[float]): array of 2 values giving the range of sensor ground truth values.
            radius (float): radius of sampling region.
        """
        area_1D = np.array([[0], [0], [0], [-radius], [radius]])
        area_2D = np.array([[0, 0], [0, radius], [0, -radius], [radius, 0], [-radius, 0]])
        super().__init__(
            noise_dev, 
            offset_function, 
            area_1D,
            area_2D,
            failure_chance,
            value_range
        )





class Thermocouple(RoundSensor):
    """
    Round sensor with a linearisation error.
    """
    def __init__(
            self, 
            temps :np.ndarray[float], 
            voltages :np.ndarray[float], 
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
        super().__init__(noise_dev, self.non_linear_error, failure_chance, value_range, radius)

    
    def non_linear_error(self, temp :float) -> float:
        """
        Calculates the voltage produced by the thermocouple.
        Then the linearised temperature measurement from the voltage.
        Then the systematic error.

        Args:
            temp (float): temperature value.

        Returns:
            float: systematic error at that temperature.
        """
        voltage = self._interpolator.predict(np.array([[temp]]))
        new_temp = self._regressor.predict(voltage)
        return new_temp - temp