import numpy as np




class Field():
    def __init__(self, regression_type :type, bounds :tuple, num_dim :int) -> None:
        self._regression_type = regression_type
        self._bounds = bounds
        self._num_dim = num_dim


    def get_bounds(self):
        return self._bounds

    
    def get_dim(self):
        return self._num_dim
    

    def fit_model(self, known_pos :np.ndarray, known_values :np.ndarray):
        pass


    def predict_values(self, pos : np.ndarray):
        pass




class ScalarField(Field):
    def __init__(self, regression_type :type, bounds :tuple, num_dim :int) -> None:
        super().__init__(regression_type, bounds, num_dim)
        self._regressor = None


    def fit_model(self, known_pos :np.ndarray, known_scalars :np.ndarray):
        self._regressor = self._regression_type(self._num_dim)
        self._regressor.fit(known_pos, known_scalars)

    
    def predict_values(self, pos):
        return self._regressor.predict(pos)




class VectorField(Field):
    def __init__(self, regression_type, bounds, num_dim) -> None:
        super().__init__(regression_type, bounds, num_dim)
        self._regressors = []


    def fit_model(self, known_pos :np.ndarray, known_vectors :np.ndarray):
        vector_dim = len(known_vectors[0])
        for i in range(vector_dim):
            regressor = self._regression_type(self._num_dim)
            regressor.fit(known_pos, known_vectors[:, i])
            self._regressors.append(regressor)


    def predict_values(self, pos):
        vector_out = []
        for regressor in self._regressors:
            vector_out.append(regressor.predict(pos))
        return np.array(vector_out).T