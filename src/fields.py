import numpy as np


class Field:
    """
    This is an abstract class to describe fields.
    They have 3 core methods.
    1. Initialisation with correct parameters.
    2. Fitting with training data to describe known values at known positions
        in the field.
    3. Predicting values at various positions in the field.
    """

    def __init__(
        self, regression_type: type, bounds: np.ndarray[float]
    ) -> None:
        """
        Args:
            regression_type (type): type of regression model.
            bounds (np.ndarray[float]): 2D array of the form [[min pos],
                [max pos]].
        """
        self._regression_type = regression_type
        self._bounds = bounds
        self._num_dim = bounds.shape[1]

    def get_bounds(self) -> np.ndarray[float]:
        """
        Returns:
            np.ndarray[float]: the bounds of the field.
        """
        return self._bounds

    def get_dim(self) -> int:
        """
        Returns:
            int: the dimensions of the field positions.
        """
        return self._num_dim


class ScalarField(Field):
    """
    Field that contains scalar values.
    """

    def __init__(self, regression_type: type, bounds: np.ndarray) -> None:
        """
        Args:
            regression_type (type): the type of regression algorithm to use to
                predict the scalar values.
            bounds (np.ndarray): the bounds of the field positions.
            num_dim (int): the number of dimensions of the field (1 or 2).
        """
        super().__init__(regression_type, bounds)
        self._regressor = None

    def fit_model(
        self, known_pos: np.ndarray, known_scalars: np.ndarray
    ) -> None:
        """
        Fit the regression model to the known field values.

        Args:
            known_pos (np.ndarray): n by d array of n positions of d
                dimensions.
            known_scalars (np.ndarray): n by 1 array of n scalar values.
        """
        self._regressor = self._regression_type(self._num_dim)
        self._regressor.fit(known_pos, known_scalars)

    def predict_values(self, pos: np.ndarray) -> np.ndarray[float]:
        """
        Predict the values at various positions in the field.

        Args:
            pos (np.ndarray): n by d array of n positions of d dimensions.

        Returns:
            np.ndarray[float]: n by 1 array of n scalars.
        """
        return self._regressor.predict(pos)


class VectorField(Field):
    def __init__(self, regression_type: type, bounds: np.ndarray) -> None:
        """
        Args:
            regression_type (type): the type of regression algorithm to use to
                predict the vector values.
            bounds (np.ndarray): the bounds of the field positions.
            num_dim (int): the number of dimensions of the field (1 or 2).
        """
        super().__init__(regression_type, bounds)
        self._regressors = []

    def fit_model(
        self, known_pos: np.ndarray, known_vectors: np.ndarray
    ) -> None:
        """
        Fit the regression model to the known field values.

        Args:
            known_pos (np.ndarray): n by d array of n positions of d
            dimensions.
            known_vectors (np.ndarray): n by m array of n vector values of m
                dimensions.
        """
        vector_dim = len(known_vectors[0])
        self._regressors = []
        for i in range(vector_dim):
            regressor = self._regression_type(self._num_dim)
            regressor.fit(known_pos, known_vectors[:, i].reshape(-1, 1))
            self._regressors.append(regressor)

    def predict_values(self, pos: np.ndarray) -> np.ndarray[float]:
        """
        Predict the values at various positions in the field.

        Args:
            pos (np.ndarray): n by d array of n positions of d dimensions.

        Returns:
            np.ndarray[float]: n by m array of n vectors of m dimensions.
        """
        vector_out = self._regressors[0].predict(pos).reshape(-1, 1)
        for regressor in self._regressors[1:]:
            new_column = regressor.predict(pos)
            vector_out = np.concatenate((vector_out, new_column), axis=1)
        return vector_out
