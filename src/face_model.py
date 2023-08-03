from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from scipy.interpolate import RBFInterpolator
from sklearn import preprocessing
import numpy as np





class GPModel():
    def __init__(self, sensor_pos, sensor_temps):
        self.__scaler = preprocessing.StandardScaler().fit(sensor_pos)
        scaled_pos = self.__scaler.transform(sensor_pos)

        self.__gp = GaussianProcessRegressor(
            kernel=RBF(), 
            n_restarts_optimizer=10, 
            normalize_y=True
        )
        self.__gp.fit(scaled_pos, sensor_temps)

    
    def get_temp(self, pos_xy):
        scaled_pos_xy = self.__scaler.transform(pos_xy.reshape(1, -1))
        return self.__gp.predict(scaled_pos_xy)[0]




class RBFModel():
    def __init__(self, sensor_pos, sensor_temps):
        self.__scaler = preprocessing.StandardScaler().fit(sensor_pos)
        scaled_pos = self.__scaler.transform(sensor_pos)

        self.__interpolater = RBFInterpolator(scaled_pos, sensor_temps)


    def get_temp(self, pos_xy):
        scaled_pos_xy = self.__scaler.transform(pos_xy.reshape(1, -1))
        return self.__interpolater(scaled_pos_xy)[0]




# class UniformGPModel():
#     def __init__(self, sensor_pos, sensor_temps):
#         y_values = sensor_pos[:, 1].reshape(-1, 1)
#         self.__scaler = preprocessing.StandardScaler().fit(y_values)
#         scaled_y_values = self.__scaler.transform(y_values)

#         kernel = RBF()
#         self.__gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10, normalize_y=True)
#         self.__gp.fit(scaled_y_values, sensor_temps)


#     def get_temp(self, pos_xy):
#         scaled_pos_xy = self.__scaler.transform(pos_xy[1].reshape(1, 1))
#         return self.__gp.predict(scaled_pos_xy)[0]
   





# class UniformRBFModel():
#     def __init__(self, sensor_pos, sensor_temps):
#         y_values = sensor_pos[:, 1].reshape(-1, 1)
#         self.__scaler = preprocessing.StandardScaler().fit(y_values)
#         scaled_y_values = self.__scaler.transform(y_values)

#         self.__interpolater = RBFInterpolator(scaled_y_values, sensor_temps)


#     def get_temp(self, pos_xy):
#         scaled_pos_xy = self.__scaler.transform(pos_xy[1].reshape(1, 1))
#         return self.__interpolater(scaled_pos_xy)[0]











