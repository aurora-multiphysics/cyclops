from scipy.interpolate import RBFInterpolator, CloughTocher2DInterpolator, CubicSpline, LinearNDInterpolator
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LinearRegression
from sklearn.gaussian_process.kernels import RBF
from sklearn import preprocessing
import numpy as np
import warnings



warnings.filterwarnings('ignore')




class RegressionModel():
    def __init__(self, num_input_dim) -> None:
        self._scaler = preprocessing.StandardScaler()
        self._regressor = None
        self._x_dim = num_input_dim


    def fit(self, train_x, train_y):
        # Note that train_x is 2D and train_y is 2D
        pass


    def predict(self, predict_x):
        # Note that predict_x is 2D and predict_y is 2D
        pass


    def check_dim(self, acceptable_dim):
        if self._x_dim not in acceptable_dim:
            raise Exception('''
                Invalid dimension of input data!
                Instead your input data should be a numpy array of shape (-1, x)
                Where x is an element of '''+str(acceptable_dim)
            )




class RBFModel(RegressionModel):
    def __init__(self, num_input_dim) -> None:
        super().__init__(num_input_dim)


    def fit(self, train_x, train_y):
        self._scaler.fit(train_x)
        scaled_x = self._scaler.transform(train_x)
        self._regressor = RBFInterpolator(scaled_x, train_y)


    def predict(self, predict_x):
        scaled_x = self._scaler.transform(predict_x)
        return self._regressor(scaled_x).reshape(-1, 1)



class LModel(RegressionModel):
    def __init__(self, num_input_dim) -> None:
        super().__init__(num_input_dim)
        self.check_dim(None)


    def check_dim(self, acceptable_dim):
        if self._x_dim <= 1:
            raise Exception('''
                Invalid dimension of input data!
                Instead your input data should be a numpy array of shape (-1, x)
                where x > 1'''
            )


    def fit(self, train_x, train_y):
        self._scaler.fit(train_x)
        scaled_x = self._scaler.transform(train_x)
        self._regressor = LinearNDInterpolator(
            scaled_x, 
            train_y,
            fill_value = np.mean(train_y)
        )


    def predict(self, predict_x):
        scaled_x = self._scaler.transform(predict_x)
        value = self._regressor(scaled_x).reshape(-1, 1)
        return value




class GPModel(RegressionModel):
    def __init__(self, num_input_dim) -> None:
        super().__init__(num_input_dim)


    def fit(self, train_x, train_y):
        self._scaler.fit(train_x)
        scaled_x = self._scaler.transform(train_x)
        self._regressor = GaussianProcessRegressor(
            kernel=RBF(), 
            n_restarts_optimizer=10, 
            normalize_y=True
        )
        self._regressor.fit(scaled_x, train_y)


    def predict(self, predict_x):
        scaled_x = self._scaler.transform(predict_x)
        return self._regressor.predict(scaled_x).reshape(-1, 1)



class PModel(RegressionModel):
    def __init__(self, num_input_dim, degree=3) -> None:
        super().__init__(num_input_dim)
        self.check_dim([1])
        self._degree = degree

    
    def fit(self, train_x, train_y):
        self._scaler.fit(train_x)
        scaled_x = self._scaler.transform(train_x)

        pos_val_matrix = np.concatenate((scaled_x, train_y.reshape(-1, 1)), axis=1)
        pos_val_matrix = pos_val_matrix[pos_val_matrix[:, 0].argsort()]

        self._regressor = np.polynomial.polynomial.Polynomial.fit(
            pos_val_matrix[:,0].reshape(-1), 
            pos_val_matrix[:,1].reshape(-1), 
            deg=self._degree
        )


    def predict(self, predict_x):
        scaled_x = self._scaler.transform(predict_x)
        return self._regressor(scaled_x).reshape(-1, 1)




class CSModel(RegressionModel):
    def __init__(self, num_input_dim) -> None:
        super().__init__(num_input_dim)
        self.check_dim([1])


    def fit(self, train_x, train_y):
        self._scaler.fit(train_x)
        scaled_x = self._scaler.transform(train_x)
        
        pos_val_matrix = np.concatenate((scaled_x, train_y.reshape(-1, 1)), axis=1)
        pos_val_matrix = pos_val_matrix[pos_val_matrix[:, 0].argsort()]

        self._regressor = CubicSpline(
            pos_val_matrix[:,0], 
            pos_val_matrix[:,1]
        )


    def predict(self, predict_x):
        scaled_x = self._scaler.transform(predict_x)
        return self._regressor(scaled_x).reshape(-1, 1)




class CTModel(RegressionModel):
    def __init__(self, num_input_dim) -> None:
        super().__init__(num_input_dim)
        self.check_dim([2])
        self._output_mean = 0


    def fit(self, train_x, train_y):
        self._scaler.fit(train_x)
        scaled_x = self._scaler.transform(train_x)
        self._regressor = CloughTocher2DInterpolator(
            scaled_x, 
            train_y,
            fill_value = np.mean(train_y)
        )


    def predict(self, predict_x):
        scaled_x = self._scaler.transform(predict_x)
        value = self._regressor(scaled_x).reshape(-1, 1)
        return value