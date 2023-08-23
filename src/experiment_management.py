import numpy as np
import pickle
import os


class Experiment():
    def __init__(self, field) -> None:
        # Todo:
        # takes in symmetry assumptions
        pass

    def design(optimiser):
        # Designs the experiment
        # Converts optimiser values into SensorSuite positions
        pass


    def generate_sensors(sensor_dict):
        sensors = []
        for sensor_name in sensor_dict.keys:
            dir_path = os.path.dirname(os.path.dirname(__file__))
            file_path = os.path.join(os.path.sep, dir_path,'sensors', sensor_name)
            sensor_file = open(file_path, 'rb')
            for i in range(sensor_dict[sensor_dict]):
                sensors.append(pickle.load(sensor_file))
            sensor_file.close()
        return sensors

class SensorSuite():
    def __init__(self, sensor_dict) -> None:
        self._sensors = self.__generate_sensors(sensor_dict)




    def set_sensors(sensor_values):
        