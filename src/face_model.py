from scipy.interpolate import RBFInterpolator, CloughTocher2DInterpolator, CubicSpline
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.exceptions import ConvergenceWarning
from sklearn.gaussian_process.kernels import RBF
from sklearn import preprocessing
import numpy as np
import warnings



warnings.filterwarnings(action='ignore', category=ConvergenceWarning)



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
        self.__extrapolater = RBFInterpolator(scaled_pos, sensor_temps)

    
    def get_temp(self, pos_xy):
        scaled_pos_xy = self.__scaler.transform(pos_xy.reshape(1, -1))
        value = self.__interpolater(scaled_pos_xy)[0]
        if value != value:
            #return 600
            return self.__extrapolater(scaled_pos_xy)[0]
        else:
            return value



class CSModel():
    def __init__(self, sensor_pos, sensor_temps):
        # IMPORTANT: Can only be used with the uniform model_manager
        self.__scaler = preprocessing.StandardScaler().fit(sensor_pos)
        scaled_pos = self.__scaler.transform(sensor_pos)

        pos_temp_matrix = np.concatenate((scaled_pos, sensor_temps.reshape(-1, 1)), axis=1)
        pos_temp_matrix = pos_temp_matrix[pos_temp_matrix[:, 0].argsort()]

        self.__cubic_spline = CubicSpline(pos_temp_matrix[:,0].reshape(-1), pos_temp_matrix[:,1].reshape(-1))


    def get_temp(self, pos_xy):
        scaled_pos_xy = self.__scaler.transform(pos_xy.reshape(1, -1))
        return self.__cubic_spline(scaled_pos_xy)[0][0]

