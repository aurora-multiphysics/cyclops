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
    1. Initialisation with correct hyperparameters.
    2. Fitting with training data.
    3. Predicting values.
    They all predict 1D outputs only.
    """
    def __init__(self, num_input_dim :int, min_length :int) -> None:
        """
        All Regression models require a scaler to rescale the input data.
        And have a regressor.
        The number of dimensions is specified to mitigate errors.

        Args:
            num_input_dim (int): number of dimensions/features of training (and test) data.
            min_length (int): minimum length of training dataset.
        """
        self._scaler = preprocessing.StandardScaler()
        self._regressor = None
        self._x_dim = num_input_dim
        self._min_length = min_length


    def prepare_fit(self, train_x :np.ndarray[float], train_y :np.ndarray[float]) -> np.ndarray[float]:
        """
        Checks that the training data is correctly dimensioned and normalises it.

        Args:
            train_x (np.ndarray[float]): n by d array of n input data values of dimension d.
            train_y (np.ndarray[float]): n by 1 array of n output data values.

        Returns:
            np.ndarray[float]: scaled n by d array of n input data values of dimension d.
        """
        self.check_dim(len(train_x[0]), self._x_dim, 'Input')
        self.check_dim(len(train_y[0]), 1, 'Output')
        self.check_length(len(train_x))
        self._scaler.fit(train_x)
        return self._scaler.transform(train_x)


    def prepare_predict(self, predict_x :np.ndarray[float]) -> np.ndarray[float]:
        """
        Checks that the prediction data is correctly dimensioned and normalises it.

        Args:
            predict_x (np.ndarray[float]): n by d array of n input data values of dimension d.

        Returns:
            np.ndarray[float]: scaled n by d array of n input data values of dimension d.
        """
        self.check_dim(len(predict_x[0]), self._x_dim, 'Input')
        return self._scaler.transform(predict_x)


    def check_dim(self, dim :int, correct_dim :int, data_name :str) -> None:
        """
        Checks the dimensions/features are equal to a specified number.

        Args:
            dim (int): measured number of dimensions.
            correct_dim (int): expected number of dimensions.
            data_name (str): name for exception handling.

        Raises:
            Exception: error to explain user's mistake.
        """
        if dim != correct_dim:
            raise Exception(data_name+' data should be a numpy array of shape (-1, '+str(correct_dim)+').')

    
    def check_length(self, length :int) -> None:
        """
        Checks the number of training data points is above a minimum length.

        Args:
            length (int): number of training data points.

        Raises:
            Exception: error to explain user's mistake.
        """
        if length < self._min_length:
            raise Exception('Input data should have a length of >= '+str(self._min_length)+'.')




class RBFModel(RegressionModel):
    """
    Uses RBF interpolation.
    Interpolates and extrapolates.
    Acts in any dimension d >= 1.
    Learns from any number of training data points n >= 2.
    Time complexity of around O(n^3).
    """
    def __init__(self, num_input_dim :int) -> None:
        super().__init__(num_input_dim, 2)
        if num_input_dim <= 0:
            raise Exception('Input data should have d >= 1 dimensions.')


    def fit(self, train_x :np.ndarray[float], train_y :np.ndarray[float]) -> None:
        """
        Fits the model to some training data.

        Args:
            train_x (np.ndarray[float]): n by d array of n training inputs with d dimensions.
            train_y (np.ndarray[float]): n by 1 array of n training outputs.
        """
        scaled_x = self.prepare_fit(train_x, train_y)
        self._regressor = RBFInterpolator(scaled_x, train_y)


    def predict(self, predict_x :np.ndarray[float]) -> np.ndarray[float]:
        """
        Returns n predicted outputs of dimension 1 given inputs.

        Args:
            predict_x (np.ndarray[float]): n by d array of n input samples of d dimensions.

        Returns:
            np.ndarray[float]: n by 1 array of n predicted 1D values.
        """
        scaled_x = self.prepare_predict(predict_x)
        return self._regressor(scaled_x).reshape(-1, 1)



class LModel(RegressionModel):
    """
    Uses linear splines.
    Only interpolates.
    Acts in any dimension d > 1.
    Learns from any number of training data points n >= 3.
    Time complexity of around O(n).
    """
    def __init__(self, num_input_dim) -> None:
        super().__init__(num_input_dim, 3)
        if num_input_dim <= 1:
            raise Exception('Input data should have d >= 2 dimensions.')


    def fit(self, train_x :np.ndarray[float], train_y :np.ndarray[float]) -> None:
        """
        Fits the model to some training data.

        Args:
            train_x (np.ndarray[float]): n by d array of n training inputs with d dimensions.
            train_y (np.ndarray[float]): n by 1 array of n training outputs.
        """
        scaled_x = self.prepare_fit(train_x, train_y)
        self._regressor = LinearNDInterpolator(
            scaled_x, 
            train_y,
            fill_value = np.mean(train_y)
        )


    def predict(self, predict_x :np.ndarray[float]) -> np.ndarray[float]:
        """
        Returns n predicted outputs of dimension 1 given inputs.

        Args:
            predict_x (np.ndarray[float]): n by d array of n input samples of d dimensions.

        Returns:
            np.ndarray[float]: n by 1 array of n predicted 1D values.
        """
        scaled_x = self.prepare_predict(predict_x)
        value = self._regressor(scaled_x).reshape(-1, 1)
        return value



class GPModel(RegressionModel):
    """
    Uses Gaussian process regression.
    Interpolates & extrapolates.
    Acts in any dimension d >= 1.
    Learns from any number of training data points n >= 3.
    Time complexity of around O(n^3).
    (also requires hyperparameter optimisation).
    """
    def __init__(self, num_input_dim :int) -> None:
        super().__init__(num_input_dim, 3)
        if num_input_dim <= 0:
            raise Exception('Input data should have d >= 1 dimensions.')


    def fit(self, train_x :np.ndarray[float], train_y :np.ndarray[float]) -> None:
        """
        Fits the model to some training data.

        Args:
            train_x (np.ndarray[float]): n by d array of n training inputs with d dimensions.
            train_y (np.ndarray[float]): n by 1 array of n training outputs.
        """
        scaled_x = self.prepare_fit(train_x, train_y)
        self._regressor = GaussianProcessRegressor(
            kernel=RBF(), 
            n_restarts_optimizer=10, 
            normalize_y=True
        )
        self._regressor.fit(scaled_x, train_y)


    def predict(self, predict_x :np.ndarray[float]) -> np.ndarray[float]:
        """
        Returns n predicted outputs of dimension 1 given inputs.

        Args:
            predict_x (np.ndarray[float]): n by d array of n input samples of d dimensions.

        Returns:
            np.ndarray[float]: n by 1 array of n predicted 1D values.
        """
        scaled_x = self.prepare_predict(predict_x)
        return self._regressor.predict(scaled_x).reshape(-1, 1)



class PModel(RegressionModel):
    """
    Uses a polynomial fit.
    Interpolates and extrapolates.
    Acts in 1D only.
    Learns from any number of training data points n >= degree.
    Time complexity of around O(n^2).
    """
    def __init__(self, num_input_dim :int, degree=3) -> None:
        super().__init__(num_input_dim, degree)
        if num_input_dim != 1:
            raise Exception('Input data should have d = 1 dimensions.')
        self._degree = degree

    
    def fit(self, train_x :np.ndarray[float], train_y :np.ndarray[float]) -> None:
        """
        Fits the model to some training data.

        Args:
            train_x (np.ndarray[float]): n by d array of n training inputs with d dimensions.
            train_y (np.ndarray[float]): n by 1 array of n training outputs.
        """
        scaled_x = self.prepare_fit(train_x, train_y)
        pos_val_matrix = np.concatenate((scaled_x, train_y.reshape(-1, 1)), axis=1)
        pos_val_matrix = pos_val_matrix[pos_val_matrix[:, 0].argsort()]

        self._regressor = np.polynomial.polynomial.Polynomial.fit(
            pos_val_matrix[:,0].reshape(-1), 
            pos_val_matrix[:,1].reshape(-1), 
            deg=self._degree
        )


    def predict(self, predict_x :np.ndarray[float]) -> np.ndarray[float]:
        """
        Returns n predicted outputs of dimension 1 given inputs.

        Args:
            predict_x (np.ndarray[float]): n by d array of n input samples of d dimensions.

        Returns:
            np.ndarray[float]: n by 1 array of n predicted 1D values.
        """
        scaled_x = self.prepare_predict(predict_x)
        return self._regressor(scaled_x).reshape(-1, 1)




class CSModel(RegressionModel):
    """
    Uses cubic spline interpolation.
    Interpolates and extrapolates.
    Acts in 1D only.
    Learns from any number of training data points n >= 2.
    Time complexity of around O(n).
    """
    def __init__(self, num_input_dim :int) -> None:
        super().__init__(num_input_dim, 2)
        if num_input_dim != 1:
            raise Exception('Input data should have d = 1 dimensions.')


    def fit(self, train_x :np.ndarray[float], train_y :np.ndarray[float]) -> None:
        """
        Fits the model to some training data.

        Args:
            train_x (np.ndarray[float]): n by d array of n training inputs with d dimensions.
            train_y (np.ndarray[float]): n by 1 array of n training outputs.
        """
        scaled_x = self.prepare_fit(train_x, train_y)
        pos_val_matrix = np.concatenate((scaled_x, train_y.reshape(-1, 1)), axis=1)
        pos_val_matrix = pos_val_matrix[pos_val_matrix[:, 0].argsort()]

        self._regressor = CubicSpline(
            pos_val_matrix[:,0], 
            pos_val_matrix[:,1]
        )


    def predict(self, predict_x :np.ndarray[float]) -> np.ndarray[float]:
        """
        Returns n predicted outputs of dimension 1 given inputs.

        Args:
            predict_x (np.ndarray[float]): n by d array of n input samples of d dimensions.

        Returns:
            np.ndarray[float]: n by 1 array of n predicted 1D values.
        """
        scaled_x = self.prepare_predict(predict_x)
        return self._regressor(scaled_x).reshape(-1, 1)




class CTModel(RegressionModel):
    """
    Uses a Clough Tocher interpolation.
    Interpolates only.
    Acts in 2D only.
    Learns from any number of training data points n >= 3.
    Time complexity of around O(n log(n)) due to the triangulation involved.
    """
    def __init__(self, num_input_dim :int) -> None:
        super().__init__(num_input_dim, 3)
        if num_input_dim != 2:
            raise Exception('Input data should have d = 2 dimensions.')
        self._output_mean = 0


    def fit(self, train_x :np.ndarray[float], train_y :np.ndarray[float]) -> None:
        """
        Fits the model to some training data.

        Args:
            train_x (np.ndarray[float]): n by d array of n training inputs with d dimensions.
            train_y (np.ndarray[float]): n by 1 array of n training outputs.
        """
        scaled_x = self.prepare_fit(train_x, train_y)
        self._regressor = CloughTocher2DInterpolator(
            scaled_x, 
            train_y,
            fill_value = np.mean(train_y)
        )


    def predict(self, predict_x :np.ndarray[float]) -> None:
        """
        Returns n predicted outputs of dimension 1 given inputs.

        Args:
            predict_x (np.ndarray[float]): n by d array of n input samples of d dimensions.

        Returns:
            np.ndarray[float]: n by 1 array of n predicted 1D values.
        """
        scaled_x = self.prepare_predict(predict_x)
        value = self._regressor(scaled_x).reshape(-1, 1)
        return value