from regressors import RBFModel, GPModel, NModel, LModel, CTModel, CSModel
from fields import ScalarField, VectorField
from read_results import PickleManager
from sensors import Sensor
import numpy as np




class SensorSuite():
    def __init__(self, field, sensor_array) -> None:
        self.__field = field
        self.__sensors = sensor_array


    def set_sensors(self, sensor_pos, sensor_values):
        measured_values = self.__set_sensor_values(sensor_values)
        print(measured_values)
        self.__field.fit_model(sensor_pos, measured_values)

    
    def __set_sensor_values(self, sensor_values):
        measured_values = np.zeros(len(sensor_values))
        for i, sensor in enumerate(self.__sensors):
            sensor.set_value(sensor_values[i])
            measured_values[i] = sensor.get_value()
        return measured_values.reshape(-1, 1)


    def predict_data(self, pos):
        return self.__field.predict_values(pos)