"""
Field classes for cyclops.

Handles scalar and vector fields for the experiment.

(c) Copyright UKAEA 2023.
"""
import numpy as np
from cyclops.sim_reader import MeshReader
from copy import deepcopy


class Field:
    """Abstract class to describe fields.

    Three core methods.
    1. Initialisation with correct parameters.
    2. Fitting with training data to describe known values at known positions
        in the field.
    3. Predicting values at various positions in the field.
    """

    @classmethod
    def mesh_reader_init(cls, mesh_reader: MeshReader, regression_type,
                         set_name=None,
                         field_2_measure=None):
        """Factory method to create a Field from a MeshReader.
        
        If no set_name is provided, it uses all points in the mesh.
        """
        # Convert set_name to list if necessary
        if set_name:
            if isinstance(set_name, str):
                set_name = [set_name]
            
            # Validate set_names
            valid_regions = mesh_reader.read_region_names()
            for s in set_name:
                invalid_sets = [s for s in set_name if s not in valid_regions]
            if invalid_sets:
                raise ValueError(f"Invalid set_name(s) provided: {invalid_sets}")

            # Read and combine positions from multiple sets
            positions = np.vstack([mesh_reader.read_pos(s) for s in set_name])
        else:
            print("Reading all mesh nodes.")
            positions = mesh_reader._MeshReader__nodes
            set_name = [None] # Treating as if single 'set'
        
        # Read and handle field data
        field_data = []

        # Check no. of field components/fields to read, should be list or str
        if isinstance(field_2_measure, list):
            # If handling ScalarField there should only be one component
            if cls != VectorField:
                raise ValueError("Only VectorFields can have multiple field "
                "components.")
            
            all_components = []
            for comp in field_2_measure:
                comp_values = []
                for s in set_name:
                    comp_values.append(mesh_reader.read_scalar(s, comp))
                all_components.append(np.concatenate(comp_values))
            field_data = np.column_stack(all_components)

        elif isinstance(field_2_measure, str):
            # If handling VectorField there should be multiple components
            if cls != ScalarField:
                raise ValueError("VectorFields should have multiple field "
                "components.")
            
            values = []
            for s in set_name:
                values.append(mesh_reader.read_scalar(s, field_2_measure))
            field_data = np.concatenate(values)

        else:
            raise ValueError("Invalid field_2_measure format. Must be a string "
            "(scalar) or list (vector).")

        # Check if field_data has been read in
        if field_data.size == 0:
            raise ValueError("field_data is empty! Input mesh appears to have "
            "no field data!")
        
        # Create an instance of the field (will compute bounds after init)
        field = cls(regression_type, positions)

        # Compute and store bounds
        field._bounds = field.get_bounds()

        # Fit the model
        field.fit_model(positions, field_data)

        return field

    def __init__(self, regression_type, bound_grid: np.ndarray) -> None:
        """Initialise class instance.

        Args:
            regression_type (type): type of regression model.
            bound_grid (np.ndarray): array of gridpoints which fully
            contains the mesh object.
        """
        if not hasattr(regression_type, "fit") or not hasattr(regression_type,
                                                            "predict"):
            raise TypeError("regression_type must be a regression model instance with fit() and predict() methods.")
        self._regression_type = regression_type
        self._bound_grid = bound_grid
        self._bounds = None  # This will be set by get_bounds()
        self._num_dim = bound_grid.shape[1]

    @staticmethod
    def safe_fit(model, X, y):
        """
        Fits any model safely, ensuring input shapes are compatible.

        Args:
            model: regression model with fit() and predict() methods.
            X (np.ndarray): input features, shape (n_samples, n_features)
            y (np.ndarray): target values, shape (n_samples,) or (n_samples, 1)
        """
        if not all(hasattr(model, method) for method in ["fit", "predict"]):
            raise TypeError("Model must implement both 'fit' and 'predict' methods.")

        X = np.atleast_2d(X)
        if X.ndim != 2:
            raise ValueError(f"X must be 2D. Got shape {X.shape}")

        y = np.atleast_2d(y)
        if y.shape[0] != X.shape[0]:
            y = y.T

        if y.shape[0] != X.shape[0]:
            raise ValueError(f"Mismatch in shapes after adjustment: X={X.shape}, y={y.shape}")

        model.fit(X, y)

    def get_bounds(self) -> np.ndarray:
        """Compute and return bounds for the field.

        Returns:
            np.ndarray: the bounds of the field.
        """
        if self._bounds is None:
            max_values = np.max(self._bound_grid, axis=0)
            min_values = np.min(self._bound_grid, axis=0)
            self._bounds = np.vstack((min_values, max_values))
        return self._bounds

    def get_dim(self) -> int:
        """Get dimensions.

        Returns:
            int: the dimensions of the field positions.
        """
        return self._num_dim

    def get_shape(self)-> np.ndarray:
        """Get shape of field"""

        return self._bound_grid.shape

class ScalarField(Field):
    """Subclass for a scalar field."""

    def __init__(self, regression_type, bounds: np.ndarray) -> None:
        """
        Initialise class instance.

        Args:
            regression_type (type): the type of regression algorithm to use to
                predict the scalar values.
            bounds (np.ndarray): the bounds of the field positions.
            num_dim (int): the number of dimensions of the field (1 or 2).
        """
        super().__init__(regression_type, bounds)
        # This will be initialised later in 'fit_model'
        self._regressor = None

    def fit_model(self, positions: np.ndarray, scalar_values: np.ndarray
    ) -> None:
        """
        Fit the regression model to the known field values.

        Args:
            positions (np.ndarray): (n, d) array of known positions.
            scalar_values (np.ndarray): (n, 1) array of known scalar
            values.
        """

        self._regressor = deepcopy(self._regression_type)
        #self._regressor.fit(positions, scalar_values)
        self.safe_fit(self._regressor, positions, scalar_values)

    def predict_values(self, pos: np.ndarray) -> np.ndarray:
        """
        Predict the values at various positions in the field.

        Args:
            pos (np.ndarray): n by d array of n positions of d dimensions.

        Returns:
            np.ndarray: n by 1 array of n scalars.
        """
        if self._regressor is None:
            raise ValueError("The regression model has not been fitted yet.")
        return self._regressor.predict(pos)
    
def get_rbf_parameters(self):
    """
    Extract RBF model parameters for symbolic reconstruction in Pyomo.

    Returns:
        centers (List[List[float]]): List of 3D coordinates used as RBF centers.
        weights (List[float]): Corresponding RBF weights.
        gamma (float): Kernel width squared (epsilon^2).
    """
    if self._regressor is None:
        raise ValueError("RBF regressor has not been fitted yet.")

    # Check if it's a RegressionModel with an RBFInterpolator inside
    inner = getattr(self._regressor, "_regressor", None)
    if inner is None or not hasattr(inner, 'coefficients'):
        raise TypeError("Regressor does not contain a compatible RBFInterpolator.")

    centers = inner.y.tolist()
    weights = inner.coefficients.tolist()
    gamma = inner.epsilon ** 2

    return centers, weights, gamma




class VectorField(Field):
    """Subclass for a vector field with regression-based interpolation."""

    def __init__(self, regression_type, bounds: np.ndarray) -> None:
        """
        Initialise class instance.

        Args:
            regression_type (type): the type of regression algorithm to use to
                predict the vector values.
            bounds (np.ndarray): the bounds of the field positions.
            num_dim (int): 
        """
        super().__init__(regression_type, bounds)
        # Each component of the vector field needs a separate regressor and
        # old regressors must be cleared inbetween
        self._regressors = []

    def fit_model(
        self, positions: np.ndarray, vector_values: np.ndarray
    ) -> None:
        """
        Fit the regression model to the known field values.true_field_vector

        Args:
            positions (np.ndarray): (n, d) array of known positions.
            vector_values (np.ndarray): (n, m) array of known vector values.
        """
        vector_dim = vector_values.shape[1]
        # Reset regressor list each time we fit
        self._regressors = []

        # Fit a regressor for each vector component
        for i in range(vector_dim):
            regressor = deepcopy(self._regression_type)
            self.safe_fit(regressor, positions, vector_values[:, i])
            inner = getattr(regressor, "_regressor", None)
            if inner is None:
                raise RuntimeError(f"[fit_model] Component {i} regressor failed to fit.")
            self._regressors.append(regressor)


    def predict_values(self, pos: np.ndarray) -> np.ndarray:
        """
        Predict the values at various positions in the field.

        Args:
            pos (np.ndarray): n by d array of n positions of d dimensions.

        Returns:
            np.ndarray: n by m array of n vectors of m dimensions.
        """
        if not self._regressors:
            raise ValueError("The model has not been fitted yet.")
        
        # Ensure pos is >1D (n by d)
        if pos.ndim == 1:
            pos = pos.reshape(1, -1)

        predictions = [reg.predict(pos) for reg in self._regressors]
        predictions = np.hstack(predictions)


        return predictions # Return the predictions as a 2D array

    def get_rbf_parameters(self):
        """
        Extract RBF model parameters for each component of a vector field.

        Returns:
            List of tuples (centers, weights, gamma) — one per vector component.
        """
        if not self._regressors:
            raise ValueError("RBF regressors have not been fitted yet.")

        all_params = []

        for i, reg in enumerate(self._regressors):
            print(f"[DEBUG] Regressor {i} outer type: {type(reg)}")
            inner = getattr(reg, "_regressor", None)
            print(f"[DEBUG] Regressor {i} inner: {inner}")
            print(f"[DEBUG] Regressor {i} inner type: {type(inner)}")

            if inner is None:
                raise TypeError(f"Regressor {i} has no fitted inner model.")

            missing = [attr for attr in ["coefficients", "y", "epsilon"] if not hasattr(inner, attr)]
            if missing:
                raise TypeError(f"Regressor {i}'s inner model is missing attributes: {missing}")

            centers = inner.y.tolist()
            weights = inner.coefficients.tolist()
            gamma = inner.epsilon ** 2

            all_params.append((centers, weights, gamma))

        return all_params



    def get_regressors(self):
        """Return the list of regressors."""
        return self._regressors

    def get_values(self, pos: np.ndarray) -> np.ndarray:
        """Return predicted values at given positions."""
        return self.predict_values(pos) 
