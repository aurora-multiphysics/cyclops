from sklearn.linear_model import LinearRegression
from scipy.interpolate import CubicSpline
import numpy as np


class Sensor():
    def __init__(self, noise_dev, offset_function) -> None:
        self._noise_dev = noise_dev
        self._offset_function = offset_function
        self._ground_truth = None

    
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
    def __init__(self, noise_dev, voltages, temps, range) -> None:
        self._regressor = LinearRegression()
        self._regressor.fit(voltages.reshape(-1, 1), temps.reshape(-1, 1))
        self._interpolator = CubicSpline(temps, voltages)
        self._range = range

        def error(self, temp):
            voltage = self._interpolator(temp)
            return self._regressor.predict(voltage.reshape(-1, 1))[0] - temp

        super().__init__(noise_dev, self.error)

    
    def set_value(self, ground_truth_array):
        ground_truth = np.mean(ground_truth_array)
        if 0 < ground_truth and ground_truth < 1370:
            self._ground_truth = ground_truth



if __name__ == '__main__':
    def f(x):
        return 0
    sensor = RoundSensor(0.1, f)
    for i in range(10):
        sensor.set_value([10, 11, 9, 12, 8])
        print(sensor.get_value())