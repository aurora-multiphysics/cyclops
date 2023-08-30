from regressors import CSModel, PModel
import numpy as np



class Sensor():
    def __init__(self, noise_dev :float, offset_function :callable, area_1D :np.ndarray[float], area_2D :np.ndarray[float], failure_chance :float, value_range :np.ndarray[float]) -> None:
        self._noise_dev = noise_dev
        self._offset_function = offset_function
        self._ground_truth = None
        self._failure_chance = failure_chance
        self._area_1D = area_1D
        self._area_2D = area_2D
        self._range = value_range

    
    def set_value(self, ground_truth_array :np.ndarray[float]) -> None:
        self._ground_truth = np.mean(ground_truth_array)

    
    def get_failure_chance(self) -> float:
        return self._failure_chance
    

    def get_value(self) -> None:
        if self._ground_truth < self._range[0]:
            return self._range[0]
        elif self._ground_truth > self._range[1]:
            return self._range[1]
        else:
            return self._ground_truth + np.random.normal()*self._noise_dev + self._offset_function(self._ground_truth)


    def get_area(self, dim :int) -> np.ndarray[float]:
        if dim == 1: 
            return self._area_1D
        else: 
            return self._area_2D

    
    def get_num_values(self) -> int:
        return len(self._area_1D)




class PointSensor(Sensor):
    def __init__(self, noise_dev :float, offset_function :callable, failure_chance :float, value_range :np.ndarray[float]) -> None:
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



class RoundSensor(Sensor):
    def __init__(self, noise_dev :float, offset_function :callable, failure_chance :float, value_range :np.ndarray[float], radius :float) -> None:
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
    def __init__(self, temps :np.ndarray[float], voltages :np.ndarray[float], noise_dev=0.6, failure_chance=0.4, radius=0.00075) -> None:
        self._regressor = PModel(1, degree=1)
        self._regressor.fit(voltages, temps)
        self._interpolator = CSModel(1)
        self._interpolator.fit(temps, voltages)

        value_range = np.array([min(temps), max(temps)])
        super().__init__(noise_dev, self.non_linear_error, failure_chance, value_range, radius)

    
    def non_linear_error(self, temp :float) -> float:
        voltage = self._interpolator.predict(np.array([[temp]]))
        new_temp = self._regressor.predict(voltage)
        return new_temp - temp


    def check_temp(self)-> bool:
        if self._ground_truth < self._range[0]:
            return False
        elif self._ground_truth > self._range[1]:
            return False
        else:
            return True