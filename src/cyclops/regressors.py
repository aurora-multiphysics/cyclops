"""
Regression classes for cyclops.

Handles the generation of predicted fields from sensor data.

(c) Copyright UKAEA 2023.
"""
import numpy as np
import holoviews as hv
hv.extension('matplotlib')
from matplotlib import pyplot as plt
from collections import deque
from scipy.interpolate import (
    RBFInterpolator,
    CloughTocher2DInterpolator,
    CubicSpline,
    LinearNDInterpolator,
    RegularGridInterpolator,
    griddata
)
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from sklearn import preprocessing

import warnings
warnings.filterwarnings("ignore")


class RegressionModel:
    """Regression model base class.

    Three core methods:
    1. Initialisation with correct hyperparameters.
    2. Fitting with training data.
    3. Predicting values.

    They all predict 1D outputs only.
    Some models can take in a variety of different input dimensions.
    """

    def __init__(self, num_input_dim: int, min_length: int) -> None:
        """Initialise class instance.

        All regression models use a scaler to rescale the input data and have
        a regressor. The number of dimensions is specified to mitigate errors.

        Args:
            num_input_dim (int): number of dimensions/features of training (and
                test) data.
            min_length (int): minimum length of training dataset.
        """
        self._scaler = preprocessing.StandardScaler()
        self._regressor = None
        self._x_dim = num_input_dim
        self._min_length = min_length

    def prepare_fit(
        self, *train_args: np.ndarray[float]) -> np.ndarray[float]:
        """Check training data dimensions #and normalise#.

        Args:
            train_x (np.ndarray[float]): n by d array of n input data values of
                dimension d.
            train_y (np.ndarray[float]): n by 1 array of n output data values.

        Returns:
            np.ndarray[float]: scaled n by d array of n input data values of
                dimension d.
        """
        
        
        
        og_shapes = []

        for dim_array in train_args[0][1:]:
            #print("heres the size", dim_array.size)
            og_s = dim_array.shape
            og_shape = dim_array.reshape(og_s)
            og_shapes.append(og_shape)
            
        for each_array, i in zip(train_args[0], range(len(train_args[0]))):
            #print(each_array)
            train_args[0][i] = each_array.flatten()
            #print(each_array.flatten())
            #print(train_args[0][i].shape)
        
        
        #scaler = preprocessing.StandardScaler().fit(X_train)
        #X_scaled = scaler.transform(X_train)
        #print("train_args[0]", train_args[0])
        scaler = self._scaler.fit(train_args[0])
        scaled_output = scaler.transform(train_args[0])
        #scale2 = self._scaler.fit(train_args[0][2:-1])
        #scaled_output2 = self._scaler.transform(train_args[0][2:-1])
        
        all_scaled_output = scaled_output#1 + scaled_output2
        #print(all_scaled_output)
        return all_scaled_output, og_shapes

    def prepare_predict(
        self, predict_x: np.ndarray[float]
    ) -> np.ndarray[float]:
        """Check prediction data dimensions and normalise.

        Args:
            predict_x (np.ndarray[float]): n by d array of n input data values
                of dimension d.

        Returns:
            np.ndarray[float]: scaled n by d array of n input data values of
                dimension d.
        """
        self.check_dim(len(predict_x[0]), self._x_dim, "Input")
        return self._scaler.transform(predict_x)

    def check_dim(self, dim: int, correct_dim: int, data_name: str) -> None:
        """Check the dimensions/features are equal to a specified number.

        Args:
            dim (int): measured number of dimensions.
            correct_dim (int): expected number of dimensions.
            data_name (str): name for exception handling.

        Raises:
            Exception: error to explain user's mistake.
        """
#        if dim != correct_dim:
#            raise Exception(
#                data_name
#                + " data should be a numpy array of shape (-1, "
#                + str(correct_dim)
#                + ")."
#            )

    def check_length(self, length: int) -> None:
        """Check the number of training data is above a minimum length.

        Args:
            length (int): number of training data points.

        Raises:
            Exception: error to explain user's mistake.
        """
        if length < self._min_length:
            raise Exception(
                "Input data should have a length of >= "
                + str(self._min_length)
                + "."
            )


class RBFModel(RegressionModel):
    """Radial basis function regressor.

    Uses RBF interpolation. Interpolates and extrapolates. Acts in any
    dimension d >= 1. Learns from any number of training data points n >= 2.
    Time complexity of around O(n^3).
    """

    def __init__(self, num_input_dim: int) -> None:
        """Initialise class instance.

        Args:
            num_input_dim (int): number of features (dimensions) for the
                training data.

        Raises:
            Exception: error to explain user's mistake.
        """
        super().__init__(num_input_dim, 2)
        if num_input_dim <= 0:
            raise Exception("Input data should have d >= 1 dimensions.")

    def fit(
        self, train_x: np.ndarray[float], train_y: np.ndarray[float]
    ) -> None:
        """Fit the model to some training data.

        Args:
            train_x (np.ndarray[float]): n by d array of n training inputs with
                d dimensions.
            train_y (np.ndarray[float]): n by 1 array of n training outputs.
        """
        scaled_x = self.prepare_fit(train_x, train_y)
        self._regressor = RBFInterpolator(scaled_x, train_y)

    def predict(self, predict_x: np.ndarray[float]) -> np.ndarray[float]:
        """Return n predicted outputs of dimension 1 given inputs.

        Args:
            predict_x (np.ndarray[float]): n by d array of n input samples of
                d dimensions.

        Returns:
            np.ndarray[float]: n by 1 array of n predicted 1D values.
        """
        scaled_x = self.prepare_predict(predict_x)
        return self._regressor(scaled_x).reshape(-1, 1)


class LModel(RegressionModel):
    """Linear regressor.

    Uses linear splines. Only interpolates. Acts in any dimension d > 1. Learns
    from any number of training data points n >= 3. Time complexity of around
    O(n).
    """

    def __init__(self, num_input_dim) -> None:
        """Initialise class instance.

        Args:
            num_input_dim (int): number of features (dimensions) for the
                training data.

        Raises:
            Exception: error to explain user's mistake.
        """
        super().__init__(num_input_dim, 3)
        if num_input_dim <= 1:
            raise Exception("Input data should have d >= 2 dimensions.")

    def fit(
        self, train_x: np.ndarray[float], train_y: np.ndarray[float]
    ) -> None:
        """Fit the model to some training data.

        Args:
            train_x (np.ndarray[float]): n by d array of n training inputs with
                d dimensions.
            train_y (np.ndarray[float]): n by 1 array of n training outputs.
        """
        scaled_x = self.prepare_fit(train_x, train_y)
        self._regressor = LinearNDInterpolator(
            scaled_x, train_y, fill_value=np.mean(train_y)
        )

    def predict(self, predict_x: np.ndarray[float]) -> np.ndarray[float]:
        """Return n predicted outputs of dimension 1 given inputs.

        Args:
            predict_x (np.ndarray[float]): n by d array of n input samples of d
                dimensions.

        Returns:
            np.ndarray[float]: n by 1 array of n predicted 1D values.
        """
        scaled_x = self.prepare_predict(predict_x)
        value = self._regressor(scaled_x).reshape(-1, 1)
        return value

#########################
class RegGridInterp(RegressionModel):
    """Multidimensional interpolation on regular or rectilinear grids.

    The data must be defined on a rectilinear grid; that is, a rectangular grid with even or uneven spacing. Linear, nearest-neighbor, spline interpolations are supported. 
    - Must update description with equivalent data to the following for it: Only interpolates. Acts in any dimension d > 1. Learns
    from any number of training data points n >= 3. Time complexity of around
    O(n).
    """

    def __init__(self, num_input_dim) -> None:
        """Initialise class instance.

        Args:
            num_input_dim (int): number of features (dimensions) for the
                training data.

        Raises:
            Exception: error to explain user's mistake.
        """
        super().__init__(num_input_dim, 3) #come back to generalise this
        #if num_input_dim <= 1:
        #    raise Exception("Input data should have d >= 2 dimensions.")

    def fit(
        self, pos_data: np.ndarray[float], field_data: np.ndarray[float]) -> None:
        """Fit the model to some training data.

        Args:
            *args (np.ndarray[float]): a series of n by d arrays containing n
            training inputs with d dimensions each. The last argument MUST be values at points described by other arguments.
        """
        
        pos_data = pos_data.T
        
        print(pos_data)
         
        x = np.array(pos_data[0][0:10])
        x = np.multipy(x, 10)
        #x = np.array([123, 134, 163, 192, 133, 150, 157, 143, 128, 137])
        y = np.array(pos_data[1][0:10])
        z = np.array(pos_data[2][0:10])
        
        data = [field_data[0:10], x, y, z]
        # We scale the data in order to avoid numerical errors when scales of different dimensions/points are very different
        scaled_data, og_shapes = self.prepare_fit(data)
        
        scaleT = np.array(scaled_data[0])
        scaleX = np.array(scaled_data[1])
        scaleY = np.array(scaled_data[2])
        scaleZ = np.array(scaled_data[3])
        
        xi,yi,zi=np.ogrid[0:1:11j, 0:1:11j, 0:1:11j]
        X1=xi.reshape(xi.shape[0],)
        Y1=yi.reshape(yi.shape[1],)
        Z1=zi.reshape(zi.shape[2],)
        
        ar_len=len(X1)*len(Y1)*len(Z1)
        
        X=np.arange(ar_len,dtype=float)
        Y=np.arange(ar_len,dtype=float)
        Z=np.arange(ar_len,dtype=float)
        
        l=0
        for i in range(0,len(X1)):
            for j in range(0,len(Y1)):
                for k in range(0,len(Z1)):
                    X[l]=X1[i]
                    Y[l]=Y1[j]
                    Z[l]=Z1[k]
                    l=l+1


        xlim = (min(x), max(x))
        ylim = (min(y), max(y))
        xgrid = np.arange(min(x), max(x), 1)
        ygrid = np.arange(min(y), max(y), 1)
        xx, yy = np.meshgrid(xgrid, ygrid, indexing='ij')
        
        x, y, z = np.meshgrid(scaleX, scaleY, scaleZ, indexing='ij')
       
        #interpolate scaled data on new grid "scaleX,scaleY,scaleZ"
        print("Interpolate...")
        #scaleintZ = griddata((scaleX,scaleY), scaleZ, (scaleX,scaleY), method='linear')
        #scaleintT = griddata((xi,yi,zi), scaleT, (xi,yi,zi), method='linear')
        print("griddata completed running...")
        #print(scaleintT)
        print(len(xgrid),len(ygrid),len(zi),len(xx),len(yy))
        zz = griddata((scaleX,scaleY), scaleZ, (xx, yy), method="linear")

        #mesh = hv.QuadMesh((xx, yy, zz))
        #img_stk = hv.ImageStack(z, bounds=(min(x), min(y), max(x), max(y)))
        #img_stk
        #contour = hv.operation.contours(mesh, levels=8)
        #scatter = hv.Scatter((x, y))
        #contour_mesh = mesh * contour * scatter
        #contour_mesh.redim(
        #    x=hv.Dimension("x", range=xlim), y=hv.Dimension("y", range=ylim),
        #) 

        X, Y, Z = np.meshgrid(scaleX, scaleY, scaleZ)

        # Create fake data
        data = scaleT

        kw = {
            'vmin': scaleT.min(),
            'vmax': scaleT.max(),
            'levels': np.linspace(data.min(), data.max(), 10),
            }

        # Create a figure with 3D ax
        fig = plt.figure(figsize=(5, 4))
        ax = fig.add_subplot(111, projection='3d')

        # Plot contour surfaces
        _ = ax.contourf(X[:, :, 0], Y[:, :, 0], data[:, :, 0],
    zdir='z', offset=0, **kw
        )
        _ = ax.contourf(
            X[0, :, :], data[0, :, :], Z[0, :, :],
            zdir='y', offset=0, **kw
        )
        C = ax.contourf(
            data[:, -1, :], Y[:, -1, :], Z[:, -1, :],
            zdir='x', offset=X.max(), **kw
        )
# --
        
        # Set limits of the plot from coord limits
        xmin, xmax = scaleX.min(), scaleX.max()
        ymin, ymax = scaleY.min(), scaleY.max()
        zmin, zmax = scaleZ.min(), scaleZ.max()
        ax.set(xlim=[xmin, xmax], ylim=[ymin, ymax], zlim=[zmin, zmax])

        # Plot edges
        edges_kw = dict(color='0.4', linewidth=1, zorder=1e3)
        ax.plot([xmax, xmax], [ymin, ymax], 0, **edges_kw)
        ax.plot([xmin, xmax], [ymin, ymin], 0, **edges_kw)
        ax.plot([xmax, xmax], [ymin, ymin], [zmin, zmax], **edges_kw)
        
        
        # Set labels and zticks
        ax.set(
        xlabel='X [km]',
        ylabel='Y [km]',
        zlabel='Z [m]')
        
        
        plt.show()
        
        #contour_mesh
        #xi,yi,zi=np.ogrid[0:1:11j, 0:1:11j, 0:1:11j]
        #print("xi", xi)
        #print("   ")
        #X1=xi.reshape(xi.shape[0],)
        #Y1=yi.reshape(yi.shape[1],)
        #Z1=zi.reshape(zi.shape[2],)
        
        #X1 = x.flatten()
        #print("X1.shape", X1.shape)
        #Y1 = y.flatten()
        #print("Y1.shape", Y1.shape)
        #Z1 = z.flatten()
        
        #ar_len=len(X1) -1
        #print("ar_len", ar_len)
        #X=np.arange(len(X1),dtype=float)
        #print("X.shape", X.shape)
        #Y=np.arange(len(Y1),dtype=float)
        #print("Y.shape", Y.shape)
        #Z=np.arange(len(Z1),dtype=float)
        #print("Z.shape", Z.shape)
        #l=0
        #for i in range(0,ar_len):
        #    for j in range(0,ar_len):
        #        for k in range(0,ar_len):
        #            X[l]=X1[i]
        #            Y[l]=Y1[j]
        #            Z[l]=Z1[k]
        #    l=l+1
        #v = temperatures
        #interpolate "data.v" on new grid "X,Y,Z"
        #print("Interpolate...")
        #V = griddata((x,y,z), v, (X,Y,Z), method='linear')
        
        #return(V, (x, y, z), (X,Y,Z))

 #       grid_x, grid_y, grid_z = np.meshgrid(x, y, z, indexing='ij')
 #       grid_temperature = temperatures[0]

        # Create a RegularGridInterpolator
  #      interpolator = RegularGridInterpolator((x, y, z), grid_temperature)

        
        #xg, yg ,zg = np.meshgrid(x, y, z, indexing='ij', sparse=True)
        #mesh_data = np.meshgrid(*args[0:-1], indexing='ij', sparse=True)
        
        #interp = RegularGridInterpolator((y,x,z), args[-1])
        

    def predict(self, predict_x: np.ndarray[float]) -> np.ndarray[float]:
        """Return n predicted outputs of dimension 1 given inputs.

        Args:
            predict_x (np.ndarray[float]): n by d array of n input samples of d
                dimensions.

        Returns:
            np.ndarray[float]: n by 1 array of n predicted 1D values.
        """
        scaled_x = self.prepare_predict(predict_x)
        value = self._regressor(scaled_x).reshape(-1, 1)
        return value
######    


class GPModel(RegressionModel):
    """Gaussian process regressor.

    Uses Gaussian process regression. Interpolates & extrapolates. Acts in any
    dimension d >= 1. Learns from any number of training data points n >= 3.
    Time complexity of around O(n^3). Requires hyperparameter optimisation.
    """

    def __init__(self, num_input_dim: int) -> None:
        """Initialise class instance.

        Args:
            num_input_dim (int): number of features (dimensions) for the
                training data.

        Raises:
            Exception: error to explain user's mistake.
        """
        super().__init__(num_input_dim, 3)
        if num_input_dim <= 0:
            raise Exception("Input data should have d >= 1 dimensions.")

    def fit(
        self, train_x: np.ndarray[float], train_y: np.ndarray[float]
    ) -> None:
        """Fit the model to some training data.

        Args:
            train_x (np.ndarray[float]): n by d array of n training inputs with
                d dimensions.
            train_y (np.ndarray[float]): n by 1 array of n training outputs.
        """
        scaled_x = self.prepare_fit(train_x, train_y)
        self._regressor = GaussianProcessRegressor(
            kernel=RBF(), n_restarts_optimizer=10, normalize_y=True
        )
        self._regressor.fit(scaled_x, train_y)

    def predict(self, predict_x: np.ndarray[float]) -> np.ndarray[float]:
        """Return n predicted outputs of dimension 1 given inputs.

        Args:
            predict_x (np.ndarray[float]): n by d array of n input samples of d
                dimensions.

        Returns:
            np.ndarray[float]: n by 1 array of n predicted 1D values.
        """
        scaled_x = self.prepare_predict(predict_x)
        return self._regressor.predict(scaled_x).reshape(-1, 1)


class PModel(RegressionModel):
    """Polynomial fit regressor.

    Uses a polynomial fit. Interpolates and extrapolates. Acts in 1D only.
    Learns from any number of training data points n >= degree. Time complexity
    of around O(n^2).
    """

    def __init__(self, num_input_dim: int, degree=3) -> None:
        """Initialise class instance.

        Args:
            num_input_dim (int): number of features (dimensions) for the
                training data.

        Raises:
            Exception: error to explain user's mistake.
        """
        super().__init__(num_input_dim, degree)
        if num_input_dim != 1:
            raise Exception("Input data should have d = 1 dimensions.")
        self._degree = degree

    def fit(
        self, train_x: np.ndarray[float], train_y: np.ndarray[float]
    ) -> None:
        """Fit the model to some training data.

        Args:
            train_x (np.ndarray[float]): n by d array of n training inputs with
                d dimensions.
            train_y (np.ndarray[float]): n by 1 array of n training outputs.
        """
        scaled_x = self.prepare_fit(train_x, train_y)
        pos_val_matrix = np.concatenate(
            (scaled_x, train_y.reshape(-1, 1)), axis=1
        )
        pos_val_matrix = pos_val_matrix[pos_val_matrix[:, 0].argsort()]

        self._regressor = np.polynomial.polynomial.Polynomial.fit(
            pos_val_matrix[:, 0].reshape(-1),
            pos_val_matrix[:, 1].reshape(-1),
            deg=self._degree,
        )

    def predict(self, predict_x: np.ndarray[float]) -> np.ndarray[float]:
        """Return n predicted outputs of dimension 1 given inputs.

        Args:
            predict_x (np.ndarray[float]): n by d array of n input samples of d
                dimensions.

        Returns:
            np.ndarray[float]: n by 1 array of n predicted 1D values.
        """
        scaled_x = self.prepare_predict(predict_x)
        return self._regressor(scaled_x).reshape(-1, 1)


class CSModel(RegressionModel):
    """Cubic spline regressor.
    
    Uses cubic spline interpolation. Interpolates and extrapolates. Acts in
    1D only. Learns from any number of training data points n >= 2. Time
    complexity of around O(n).
    """

    def __init__(self, num_input_dim: int) -> None:
        """Initialise class instance.

        Args:
            num_input_dim (int): number of features (dimensions) for the
                training data.

        Raises:
            Exception: error to explain user's mistake.
        """
        super().__init__(num_input_dim, 2)
        if num_input_dim != 1:
            raise Exception("Input data should have d = 1 dimensions.")

    def fit(
        self, train_x: np.ndarray[float], train_y: np.ndarray[float]
    ) -> None:
        """Fit the model to some training data.

        Args:
            train_x (np.ndarray[float]): n by d array of n training inputs with
                d dimensions.
            train_y (np.ndarray[float]): n by 1 array of n training outputs.
        """
        scaled_x = self.prepare_fit(train_x, train_y)
        pos_val_matrix = np.concatenate(
            (scaled_x, train_y.reshape(-1, 1)), axis=1
        )
        pos_val_matrix = pos_val_matrix[pos_val_matrix[:, 0].argsort()]

        self._regressor = CubicSpline(
            pos_val_matrix[:, 0], pos_val_matrix[:, 1]
        )

    def predict(self, predict_x: np.ndarray[float]) -> np.ndarray[float]:
        """Return n predicted outputs of dimension 1 given inputs.

        Args:
            predict_x (np.ndarray[float]): n by d array of n input samples of d
                dimensions.

        Returns:
            np.ndarray[float]: n by 1 array of n predicted 1D values.
        """
        scaled_x = self.prepare_predict(predict_x)
        return self._regressor(scaled_x).reshape(-1, 1)


class CTModel(RegressionModel):
    """Clough Tocher regressor.

    Uses a Clough Tocher interpolation. Interpolates only. Acts in 2D only,
    Learns from any number of training data points n >= 3. Time complexity of
    around O(n log(n)) due to the triangulation involved.
    """

    def __init__(self, num_input_dim: int) -> None:
        """Initialise class instance.

        Args:
            num_input_dim (int): number of features (dimensions) for the
                training data.

        Raises:
            Exception: error to explain user's mistake.
        """
        super().__init__(num_input_dim, 3)
        if num_input_dim != 2:
            raise Exception("Input data should have d = 2 dimensions.")
        self._output_mean = 0

    def fit(
        self, train_x: np.ndarray[float], train_y: np.ndarray[float]
    ) -> None:
        """Fit the model to some training data.

        Args:
            train_x (np.ndarray[float]): n by d array of n training inputs with
                d dimensions.
            train_y (np.ndarray[float]): n by 1 array of n training outputs.
        """
        scaled_x = self.prepare_fit(train_x, train_y)
        self._regressor = CloughTocher2DInterpolator(
            scaled_x, train_y, fill_value=np.mean(train_y)
        )

    def predict(self, predict_x: np.ndarray[float]) -> None:
        """Return n predicted outputs of dimension 1 given inputs.

        Args:
            predict_x (np.ndarray[float]): n by d array of n input samples of d
                dimensions.

        Returns:
            np.ndarray[float]: n by 1 array of n predicted 1D values.
        """
        scaled_x = self.prepare_predict(predict_x)
        value = self._regressor(scaled_x).reshape(-1, 1)
        return value
