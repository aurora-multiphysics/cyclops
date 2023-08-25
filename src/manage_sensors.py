from sensors import Sensor
from fields import Field
import numpy as np




class SensorSuite():
    def __init__(self, field :Field, sensors :list[Sensor]) -> None:
        self.__field = field
        self.__sensors = sensors


    def set_sensors(self, sensor_pos :np.ndarray, sensor_values :np.ndarray):
        measured_values = self.__set_sensor_values(sensor_values)
        self.__field.fit_model(sensor_pos, measured_values)

    
    def __set_sensor_values(self, sensor_values :np.ndarray):
        measured_values = np.zeros(sensor_values.shape)
        for i, sensor in enumerate(self.__sensors):
            sensor.set_value(sensor_values[i, 0])
            measured_values[i, 0] = sensor.get_value()
        return measured_values


    def predict_data(self, pos):
        return self.__field.predict_values(pos)