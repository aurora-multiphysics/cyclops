from results_management import PickleManager
from regressors import GPModel, RBFModel
from graph_management import GraphManager
import numpy as np


class Experiment():
    def __init__(self, region, sensor_suite) -> None:
        self.__region = region
        self.__comparison_pos = region.get_pos()
        self.__comparison_values = region.get_values()
        self.__sensor_suite = sensor_suite






class SensorSuite():
    def __init__(self, sensors, regressor_type=RBFModel) -> None:
        self.__sensors = sensors
        self.__sensor_pos = None
        self.__regressor = None
        self.__regressor_type = regressor_type
    

    def set_model(self, sensor_pos, sensor_values):
        self.__sensor_pos = sensor_pos
        self.__set_sensor_values(sensor_values)
        self.__regressor = self.__regressor_type(sensor_pos, self.get_sensor_values())

    
    def get_field_value(self, pos):
        return self.__regressor.get_value(pos)

    
    def get_sensor_pos(self):
        return self.__sensor_pos

    
    def __set_sensor_values(self, sensor_values):
        for i, value in enumerate(sensor_values):
            self.__sensors[i].set_value(value)
    

    def get_sensor_values(self):
        values = []
        for sensor in self.__sensors:
            values.append(sensor.get_value())
        return values



if __name__ == '__main__':
    pickle_manager = PickleManager()
    sensors = []
    for i in range(5):
        sensors.append(pickle_manager.read_file('sensors', 'k-type.objs'))
    sensor_suite = SensorSuite(sensors)

    sensor_suite.set_model([[0.01], [0.01], [0.01]])