from scipy.interpolate import RBFInterpolator, CloughTocher2DInterpolator, CubicSpline
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.exceptions import ConvergenceWarning
from sklearn.gaussian_process.kernels import RBF
from sklearn import preprocessing
import numpy as np
import warnings



warnings.filterwarnings(action='ignore', category=ConvergenceWarning)
warnings.filterwarnings(action='ignore', category=np.RankWarning)



class ScalarRegressionModel():
    def __init__(self, sensor_pos):
        self._scaler = preprocessing.StandardScaler()
        self._scaler.fit(sensor_pos)



class GPModel(ScalarRegressionModel):
    def __init__(self, sensor_pos, sensor_values):
        super().__init__(sensor_pos)
        scaled_pos = self._scaler.transform(sensor_pos)

        self.__interpolator = GaussianProcessRegressor(
            kernel=RBF(), 
            n_restarts_optimizer=10, 
            normalize_y=True
        )
        self.__interpolator.fit(scaled_pos, sensor_values)


    def get_temp(self, pos_xy):
        scaled_pos_xy = self._scaler.transform(pos_xy)
        return self.__interpolator.predict(scaled_pos_xy)




class RBFModel(ScalarRegressionModel):
    def __init__(self, sensor_pos, sensor_values):
        super().__init__(sensor_pos)
        scaled_pos = self._scaler.transform(sensor_pos)
        self.__interpolater = RBFInterpolator(scaled_pos, sensor_values)


    def get_temp(self, pos_xy):
        scaled_pos_xy = self.__scaler.transform(pos_xy)
        return self.__interpolater(scaled_pos_xy)