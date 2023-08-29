from sensors import Sensor
from fields import Field
import numpy as np



class SymmetryManager():
    def __init__(self) -> None:
        self.__x_point = 0
        self.__x_line = 0
        self.__y_line = 0
        self.__grad = 0

    
    def set_1D_x(self, value):
        self.__x_point = value

    
    def set_2D_x(self, value):
        self.__x_line = value

    
    def set_2D_y(self, value):
        self.__y_line = value


    def set_2D_grad(self, value):
        self.__grad = value


    def reflect_1D(self, x_pos):
        axis = np.ones(x_pos.shape)*self.__x_point
        reflected_arr = 2*axis - x_pos
        return np.concatenate((x_pos, reflected_arr), axis=0)


    def reflect_2D_horiz(self, pos):
        axis = np.ones(len(pos))*self.__x_line
        reflected_arr = np.copy(pos)
        reflected_arr[:, 0] = 2*axis - pos[:, 0]
        return np.concatenate((pos, reflected_arr), axis=0)


    def reflect_2D_vert(self, pos):
        axis = np.ones(len(pos))*self.__y_line
        reflected_arr = np.copy(pos)
        reflected_arr[:, 1] = 2*axis - pos[:, 1]
        return np.concatenate((pos, reflected_arr), axis=0)


    def reflect_2D_line(self, pos):
        m = self.__grad
        reflect_matrix = 1/(1+m**2)*np.array([[1 - m**2, 2*m], [2*m, m**2 - 1]])
        reflected_arr = np.apply_along_axis(reflect_matrix.dot, 0, pos.T)
        return np.concatenate((pos, reflected_arr.T), axis=0)




class SensorSuite():
    def __init__(self, field :Field, sensors :list[Sensor], symmetry=[]) -> None:
        self.__field = field
        self.__sensors = sensors
        self.__num_sensors = len(self.__sensors)
        self.__symmetry = symmetry
        self.__bounds = field.get_bounds()
        self.__num_dim = field.get_dim()
        

    def get_predict_pos(self, sensor_pos :np.array):
        record_pos = []
        for i, sensor in enumerate(self.__sensors):
            area = sensor.get_area(self.__num_dim)
            positions = sensor_pos[i]*np.ones(area.shape) + area
            for pos in positions:
                record_pos.append(pos)
        return np.array(record_pos)


    def fit_sensor_model(self, sensor_pos :np.ndarray, measured_values :np.ndarray):
        for transformation in self.__symmetry:
            sensor_pos = transformation(sensor_pos)
            measured_values = np.concatenate((measured_values, measured_values), axis=0)
        new_pos, new_values = self.filter(sensor_pos, measured_values)
        self.__field.fit_model(new_pos, new_values)


    def get_num_sensors(self):
        return self.__num_sensors

    
    def set_sensor_values(self, sensor_values :np.ndarray):
        measured_values = np.zeros(self.__num_sensors).reshape(-1, 1)
        index = 0
        for i, sensor in enumerate(self.__sensors):
            relevant_values = sensor_values[index:index+sensor.get_num_values(), 0]
            index += sensor.get_num_values()
            sensor.set_value(relevant_values)
            measured_values[i, 0] = sensor.get_value()
        return measured_values


    def predict_data(self, pos):
        return self.__field.predict_values(pos)

    
    def filter(self, pos_array, measured_values):
        out_pos = []
        out_value = []
        condition_1 = pos_array > self.__bounds[0]
        condition_2 = pos_array < self.__bounds[1]
        for i, pos in enumerate(pos_array):
            if condition_1[i] == True and condition_2[i] == True:
                out_pos.append(pos_array[i])
                out_value.append(measured_values[i])
        return np.array(out_pos), np.array(out_value)