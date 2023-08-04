from scipy.interpolate import RBFInterpolator, CloughTocher2DInterpolator
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
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




class CTModel():
    def __init__(self, sensor_pos, sensor_temps):
        # IMPORTANT: Can only be used with the symmetric model_manager
        self.__scaler = preprocessing.StandardScaler().fit(sensor_pos)
        scaled_pos = self.__scaler.transform(sensor_pos)

        self.__interpolater = CloughTocher2DInterpolator(scaled_pos, sensor_temps)

    
    def get_temp(self, pos_xy):
        scaled_pos_xy = self.__scaler.transform(pos_xy.reshape(1, -1))
        value = self.__interpolater(scaled_pos_xy)[0]
        if value != value:
            return 600
        else:
            return value




class CTRBFModel():
    def __init__(self, sensor_pos, sensor_temps):
        # IMPORTANT: Can only be used with the symmetric model_manager
        self.__scaler = preprocessing.StandardScaler().fit(sensor_pos)
        scaled_pos = self.__scaler.transform(sensor_pos)

        self.__interpolater = CloughTocher2DInterpolator(scaled_pos, sensor_temps)
        self.__extrapolater = RBFInterpolator(scaled_pos, sensor_temps)

    
    def get_temp(self, pos_xy):
        scaled_pos_xy = self.__scaler.transform(pos_xy.reshape(1, -1))
        value = self.__interpolater(scaled_pos_xy)[0]
        if value != value:
            return self.__extrapolater(scaled_pos_xy)[0]
        else:
            return value


