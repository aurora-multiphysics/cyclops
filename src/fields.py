"""
Defines the ScalarField and VectorField objects.
"""
import numpy as np




class Field():
    """
    This is an abstract class to describe fields.
    They have 3 core methods.
    1. initialisation with correct parameters
    2. fitting with training data to describe known values at known positions in the field
    3. predicting values at various positions in the field
    """
    def __init__(self, regression_type :type, bounds :np.ndarray[float], num_dim :int) -> None:
        self._regression_type = regression_type
        self._bounds = bounds
        self._num_dim = num_dim


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
    

    def fit_model(self, known_pos :np.ndarray, known_values :np.ndarray) -> None:
        """
        Fits a regression model to the positions and values given.

        Args:
            known_pos (np.ndarray): contains n positions of d dimensions where d is _num_dim
            known_values (np.ndarray): contains n values of m dimensions (where m may be any natural number)
        """
        pass


    def predict_values(self, pos : np.ndarray) -> np.ndarray[float]:
        """
        Predicts the values of the field at the positions specified.

        Args:
            pos (np.ndarray): n by d array of n positions of d dimensions where d is _num_dim

        Returns:
            np.ndarray[float]: n by m array of n values of m dimensions where m may be any real number
        """
        pass




class ScalarField(Field):
    def __init__(self, regression_type :type, bounds :np.ndarray, num_dim :int) -> None:
        super().__init__(regression_type, bounds, num_dim)
        self._regressor = None


    def fit_model(self, known_pos :np.ndarray, known_scalars :np.ndarray) -> None:
        self._regressor = self._regression_type(self._num_dim)
        self._regressor.fit(known_pos, known_scalars)

    
    def predict_values(self, pos :np.ndarray) -> np.ndarray[float]:
        return self._regressor.predict(pos)




class VectorField(Field):
    def __init__(self, regression_type :type, bounds :np.ndarray, num_dim :int) -> None:
        super().__init__(regression_type, bounds, num_dim)
        self._regressors = []


    def fit_model(self, known_pos :np.ndarray, known_vectors :np.ndarray) -> None:
        vector_dim = len(known_vectors[0])
        for i in range(vector_dim):
            regressor = self._regression_type(self._num_dim)
            regressor.fit(known_pos, known_vectors[:, i])
            self._regressors.append(regressor)


    def predict_values(self, pos :np.ndarray) -> np.ndarray[float]:
        vector_out = self._regressors[0].predict(pos).reshape(-1, 1)
        for regressor in self._regressors[1:]:
            new_column = regressor.predict(pos)
            vector_out = np.concatenate((vector_out, new_column), axis=1)
        return vector_out