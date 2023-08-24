from sklearn.linear_model import LinearRegression
from scipy.interpolate import CubicSpline
import numpy as np



class Sensor():
    id_num = 0
    def __init__(self, noise_dev, offset_function) -> None:
        self._noise_dev = noise_dev
        self._offset_function = offset_function
        self._ground_truth = None
        self._id_num = self.id_num
        self.id_num += 1

    
    def get_id(self):
        return self._id_num

    
    def set_value(self, ground_truth):
        self._ground_truth = ground_truth
    

    def get_value(self):
        return self._ground_truth + np.random.normal()*self._noise_dev + self._offset_function(self._ground_truth)





class PointSensor(Sensor):
    def __init__(self, noise_dev, offset_function) -> None:
        super().__init__(noise_dev, offset_function)





class RoundSensor(Sensor):
    def __init__(self, noise_dev, offset_function) -> None:
        super().__init__(noise_dev, offset_function)

    
    def set_value(self, ground_truth_array):
        self._ground_truth = np.mean(ground_truth_array)





class Thermocouple(RoundSensor):
    def __init__(self, temps, voltages, noise_dev=0.6, failure_chance=0.4) -> None:
        self._regressor = LinearRegression()
        self._regressor.fit(voltages.reshape(-1, 1), temps.reshape(-1, 1))
        self._interpolator = CubicSpline(temps, voltages)
        self._failure_chance = failure_chance

        self._range = (min(temps), max(temps))
        super().__init__(noise_dev, self.non_linear_error)

    
    def non_linear_error(self, temp):
        voltage = self._interpolator(temp)
        new_temp = self._regressor(voltage)
        return new_temp - temp


    def check_temp(self):
        if self._ground_truth < self._range[0]:
            return False
        elif self._ground_truth > self._range[1]:
            return False
        else:
            return True

    
    def get_failure_chance(self):
        return self._failure_chance