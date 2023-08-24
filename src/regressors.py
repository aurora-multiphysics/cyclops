from scipy.interpolate import RBFInterpolator, CloughTocher2DInterpolator, CubicSpline, LinearNDInterpolator, NearestNDInterpolator
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.exceptions import ConvergenceWarning
from sklearn.gaussian_process.kernels import RBF
from sklearn import preprocessing
import numpy as np
import warnings

from matplotlib import pyplot as plt

warnings.filterwarnings(action='ignore', category=ConvergenceWarning)
warnings.filterwarnings(action='ignore', category=np.RankWarning)



class RegressionModel():
    def __init__(self) -> None:
        self._scaler = preprocessing.StandardScaler()
        self._regressor = None


    def fit(self, train_x, train_y):
        # Note that train_x is 2D and train_y is 2D
        pass


    def predict(self, predict_x):
        # Note that predict_x is 2D and predict_y is 2D
        pass




class RBFModel(RegressionModel):
    def __init__(self) -> None:
        # No restrictions!
        super().__init__()


    def fit(self, train_x, train_y):
        self._scaler.fit(train_x)
        scaled_x = self._scaler.transform(train_x)
        self._regressor = RBFInterpolator(scaled_x, train_y)


    def predict(self, predict_x):
        scaled_x = self._scaler.transform(predict_x)
        return self._regressor(scaled_x)




class NModel(RegressionModel):
    def __init__(self) -> None:
        # No restrictions!
        super().__init__()


    def fit(self, train_x, train_y):
        self._scaler.fit(train_x)
        scaled_x = self._scaler.transform(train_x)
        self._regressor = NearestNDInterpolator(scaled_x, train_y)


    def predict(self, predict_x):
        scaled_x = self._scaler.transform(predict_x)
        return self._regressor(scaled_x)



class LModel(RegressionModel):
    def __init__(self) -> None:
        # Only for > 1D inputs
        # Only for interpolation
        super().__init__()


    def fit(self, train_x, train_y):
        # Note that train_x is 2D and train_y is 2D
        self._scaler.fit(train_x)
        scaled_x = self._scaler.transform(train_x)
        self._regressor = LinearNDInterpolator(scaled_x, train_y)


    def predict(self, predict_x):
        # Note that predict_x is 2D and predict_y is 2D
        scaled_x = self._scaler.transform(predict_x)
        return self._regressor(scaled_x)    




class GPModel(RegressionModel):
    def __init__(self) -> None:
        super().__init__()


    def fit(self, train_x, train_y):
        # Note that train_x is 2D and train_y is 2D
        self._scaler.fit(train_x)
        scaled_x = self._scaler.transform(train_x)
        self._regressor = GaussianProcessRegressor(
            kernel=RBF(), 
            n_restarts_optimizer=10, 
            normalize_y=True
        )
        self._regressor.fit(scaled_x, train_y)


    def predict(self, predict_x):
        # Note that predict_x is 2D and predict_y is 2D
        scaled_x = self._scaler.transform(predict_x)
        return self._regressor.predict(scaled_x)




class CSModel(RegressionModel):
    def __init__(self) -> None:
        # Only for 1D inputs
        super().__init__()


    def fit(self, train_x, train_y):
        # Note that train_x is 2D and train_y is 2D
        self._scaler.fit(train_x)
        scaled_x = self._scaler.transform(train_x)
        self._regressor = CubicSpline(scaled_x.reshape(-1), train_y.reshape(-1))


    def predict(self, predict_x):
        # Note that predict_x is 2D and predict_y is 2D
        scaled_x = self._scaler.transform(predict_x)
        return self._regressor(scaled_x).reshape(-1, 1)




class CTModel(RegressionModel):
    def __init__(self) -> None:
        # Only for 2D inputs
        # Only for interpolation
        super().__init__()


    def fit(self, train_x, train_y):
        # Note that train_x is 2D and train_y is 2D
        self._scaler.fit(train_x)
        scaled_x = self._scaler.transform(train_x)
        self._regressor = CloughTocher2DInterpolator(scaled_x, train_y)


    def predict(self, predict_x):
        # Note that predict_x is 2D and predict_y is 2D
        scaled_x = self._scaler.transform(predict_x)
        return self._regressor(scaled_x).reshape(-1, 1)