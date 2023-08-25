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


    def set_sensors(self, sensor_pos :np.ndarray, sensor_values :np.ndarray):
        measured_values = self.__set_sensor_values(sensor_values)
        for transformation in self.__symmetry:
            sensor_pos = transformation(sensor_pos)
            measured_values = np.concatenate((measured_values, measured_values), axis=0)
        self.filter(sensor_pos, measured_values)
        self.__field.fit_model(sensor_pos, measured_values)


    def get_num_sensors(self):
        return self.__num_sensors

    
    def __set_sensor_values(self, sensor_values :np.ndarray):
        measured_values = np.zeros(sensor_values.shape)
        for i, sensor in enumerate(self.__sensors):
            sensor.set_value(sensor_values[i, 0])
            measured_values[i, 0] = sensor.get_value()
        return measured_values


    def predict_data(self, pos):
        return self.__field.predict_values(pos)

    
    def filter(self, pos_array, measured_values):
        joint_array = np.concatenate((pos_array, measured_values), axis=1)
        joint_array = np.argwhere(pos_array[:,0:self.__num_dim] > self.__bounds[0])
        joint_array = np.argwhere(pos_array[:,0:self.__num_dim] < self.__bounds[1])
        return joint_array[:,0:self.__num_dim], joint_array[:,-1].reshape(-1, 1)