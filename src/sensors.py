import numpy as np
import pickle
import os




class Sensor():
    def __init__(self, random_error, linear_error) -> None:
        self._noise_dev = random_error
        self._error_function = linear_error
        self._ground_truth = None

    
    def set_value(self, ground_truth):
        self._ground_truth = ground_truth
    

    def get_value(self):
        linear_temp = self._error_function(self._ground_truth)
        return linear_temp + np.random.normal()*self._noise_dev




class PointSensor(Sensor):
    def __init__(self, random_error, error_function) -> None:
        super().__init__(random_error, error_function)




class RoundSensor(Sensor):
    def __init__(self, random_error, error_function) -> None:
        super().__init__(random_error, error_function)

    
    def set_value(self, ground_truth_array):
        self._ground_truth = np.mean(ground_truth_array)





class Thermocouple(RoundSensor):
    def __init__(self, random_error, file_name) -> None:
        dir_path = os.path.dirname(os.path.dirname(__file__))
        full_path = os.path.join(os.path.sep, dir_path,'sensors', file_name)
        
        error_file = open(full_path, 'rb')
        error_object = pickle.load(error_file)
        error_file.close()
        super().__init__(random_error, error_object.error_function)


