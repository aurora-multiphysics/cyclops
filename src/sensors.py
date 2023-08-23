from sklearn.linear_model import LinearRegression
from scipy.interpolate import CubicSpline
import pandas as pd
import numpy as np
import pickle
import os




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
    def __init__(self, csv_name) -> None:
        super().__init__(
            0.6, 
            self._generate_offset_function(csv_name)
        )


    def _generate_offset_function(self, csv_name):
        dir_path = os.path.dirname(os.path.dirname(__file__)) 
        full_path = os.path.join(os.path.sep, dir_path,'sensors', csv_name)
        dataframe = pd.read_csv(full_path)

        temps = dataframe['T'].values
        voltages = dataframe['V'].values

        regressor = LinearRegression()
        regressor.fit(voltages.reshape(-1, 1), temps.reshape(-1, 1))
        interpolator = CubicSpline(temps, voltages)

        def offset(temp):
            voltage = interpolator(temp)
            new_temp = regressor(voltage)
            return new_temp - temp
        
        return offset


if __name__ == '__main__':
    k_type = Thermocouple('k-type.csv')
    sensor_file = open('k-type.obj', 'wb')
    pickle.dump(k_type, sensor_file)
    sensor_file.close()