"""
This modules contains the various regression models. 
"""
from scipy.interpolate import RBFInterpolator, CloughTocher2DInterpolator, CubicSpline, LinearNDInterpolator
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from sklearn import preprocessing
import numpy as np
import warnings



warnings.filterwarnings('ignore')




class RegressionModel():
    """
    This is an abstract class to describe regression models.
    They have 3 core methods.
    1. initialisation with correct hyperparameters
    2. fitting with training data
    3. predicting values
    They all predict 1D outputs only so many are required to predict a vector.
    """
    def __init__(self, num_input_dim :int) -> None:
        """
        All Regression models require a scaler to rescale the input data.
        And have a regressor.
        The number of dimensions is specified to mitigate errors.

        Args:
            num_input_dim (int): number of dimensions of an input sample.
        """
        self._scaler = preprocessing.StandardScaler()
        self._regressor = None
        self._x_dim = num_input_dim


    def fit(self, train_x :np.ndarray[float], train_y :np.ndarray[float]) -> None:
        """
        Fits the model to some training data.

        Args:
            train_x (np.ndarray[float]): n by d array of n training inputs with d dimensions
            train_y (np.ndarray[float]): n by 1 array of n training outputs
        """
        pass


    def predict(self, predict_x :np.ndarray[float]) -> None:
        """
        Returns n predicted outputs of dimension 1 given inputs

        Args:
            predict_x (np.ndarray[float]): n by d array of n input samples of d dimension
        """
        pass


    def check_dim(self, acceptable_dim :set) -> None:
        """
        Ensures that the number of dimensions specified is one the model can cope with

        Args:
            acceptable_dim (set): the dimensions the model can cope with.

        Raises:
            Exception: an error if the model cannot cope.
        """
        if self._x_dim not in acceptable_dim:
            raise Exception('''
                Invalid dimension of input data!
                Instead your input data should be a numpy array of shape (-1, x)
                Where x is an element of '''+str(acceptable_dim)
            )




class RBFModel(RegressionModel):
    """
    Uses RBF interpolation.
    Interpolates and extrapolates.
    Works in any dimensions.
    It describes the n training data points as a sum of n RBF functions.
    Hence time complexity of around O(n^3).
    """
    def __init__(self, num_input_dim :int) -> None:
        super().__init__(num_input_dim)


    def fit(self, train_x :np.ndarray[float], train_y :np.ndarray[float]) -> None:
        self._scaler.fit(train_x)
        scaled_x = self._scaler.transform(train_x)
        self._regressor = RBFInterpolator(scaled_x, train_y)


    def predict(self, predict_x :np.ndarray[float]) -> np.ndarray[float]:
        scaled_x = self._scaler.transform(predict_x)
        return self._regressor(scaled_x).reshape(-1, 1)



class LModel(RegressionModel):
    """
    Uses linear splines.
    Only interpolates.
    Only works in >1 dimensions.
    Time complexity of around O(n).
    """
    def __init__(self, num_input_dim) -> None:
        super().__init__(num_input_dim)
        self.check_dim({})


    def check_dim(self, acceptable_dim :set) -> None:
        if self._x_dim <= 1:
            raise Exception('''
                Invalid dimension of input data!
                Instead your input data should be a numpy array of shape (-1, x)
                where x > 1'''
            )


    def fit(self, train_x :np.ndarray[float], train_y :np.ndarray[float]) -> None:
        self._scaler.fit(train_x)
        scaled_x = self._scaler.transform(train_x)
        self._regressor = LinearNDInterpolator(
            scaled_x, 
            train_y,
            fill_value = np.mean(train_y)
        )


    def predict(self, predict_x :np.ndarray[float]) -> np.ndarray[float]:
        scaled_x = self._scaler.transform(predict_x)
        value = self._regressor(scaled_x).reshape(-1, 1)
        return value




class GPModel(RegressionModel):
    """
    Uses Gaussian process regression.
    Interpolates & extrapolates.
    Works in any dimensions.
    Time complexity of around O(n^3).
    (also requires hyperparameter optimisation).
    """
    def __init__(self, num_input_dim :int) -> None:
        super().__init__(num_input_dim)


    def fit(self, train_x :np.ndarray[float], train_y :np.ndarray[float]) -> None:
        self._scaler.fit(train_x)
        scaled_x = self._scaler.transform(train_x)
        self._regressor = GaussianProcessRegressor(
            kernel=RBF(), 
            n_restarts_optimizer=10, 
            normalize_y=True
        )
        self._regressor.fit(scaled_x, train_y)


    def predict(self, predict_x :np.ndarray[float]) -> np.ndarray[float]:
        scaled_x = self._scaler.transform(predict_x)
        return self._regressor.predict(scaled_x).reshape(-1, 1)



class PModel(RegressionModel):
    """
    Uses a polynomial fit.
    Interpolates and extrapolates.
    Only works in 1D.
    Time complexity of around O(n^2).
    """
    def __init__(self, num_input_dim :int, degree=3) -> None:
        super().__init__(num_input_dim)
        self.check_dim({1})
        self._degree = degree

    
    def fit(self, train_x :np.ndarray[float], train_y :np.ndarray[float]) -> None:
        self._scaler.fit(train_x)
        scaled_x = self._scaler.transform(train_x)

        pos_val_matrix = np.concatenate((scaled_x, train_y.reshape(-1, 1)), axis=1)
        pos_val_matrix = pos_val_matrix[pos_val_matrix[:, 0].argsort()]

        self._regressor = np.polynomial.polynomial.Polynomial.fit(
            pos_val_matrix[:,0].reshape(-1), 
            pos_val_matrix[:,1].reshape(-1), 
            deg=self._degree
        )


    def predict(self, predict_x :np.ndarray[float]) -> np.ndarray[float]:
        scaled_x = self._scaler.transform(predict_x)
        return self._regressor(scaled_x).reshape(-1, 1)




class CSModel(RegressionModel):
    """
    Uses cubic spline interpolation.
    Interpolates and extrapolates.
    Only works in 1D.
    Time complexity of around O(n).
    """
    def __init__(self, num_input_dim :int) -> None:
        super().__init__(num_input_dim)
        self.check_dim({1})


    def fit(self, train_x :np.ndarray[float], train_y :np.ndarray[float]) -> None:
        self._scaler.fit(train_x)
        scaled_x = self._scaler.transform(train_x)
        
        pos_val_matrix = np.concatenate((scaled_x, train_y.reshape(-1, 1)), axis=1)
        pos_val_matrix = pos_val_matrix[pos_val_matrix[:, 0].argsort()]

        self._regressor = CubicSpline(
            pos_val_matrix[:,0], 
            pos_val_matrix[:,1]
        )


    def predict(self, predict_x :np.ndarray[float]) -> np.ndarray[float]:
        scaled_x = self._scaler.transform(predict_x)
        return self._regressor(scaled_x).reshape(-1, 1)




class CTModel(RegressionModel):
    """
    Uses a Clough Tocher interpolation.
    Interpolates only.
    Only works in 2D.
    Time complexity of around O(n log(n)) due to the triangulation involved.
    """
    def __init__(self, num_input_dim :int) -> None:
        super().__init__(num_input_dim)
        self.check_dim([2])
        self._output_mean = 0


    def fit(self, train_x :np.ndarray[float], train_y :np.ndarray[float]) -> None:
        self._scaler.fit(train_x)
        scaled_x = self._scaler.transform(train_x)
        self._regressor = CloughTocher2DInterpolator(
            scaled_x, 
            train_y,
            fill_value = np.mean(train_y)
        )


    def predict(self, predict_x :np.ndarray[float]) -> None:
        scaled_x = self._scaler.transform(predict_x)
        value = self._regressor(scaled_x).reshape(-1, 1)
        return value