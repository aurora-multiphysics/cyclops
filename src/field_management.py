import numpy as np




class Field():
    def __init__(self, regression_type, field_bounds) -> None:
        self._regression_type = regression_type
        self._bounds = field_bounds


    def get_bounds(self):
        return self._bounds
    

    def fit_model(self, known_pos, known_values):
        pass


    def predict_value(self, pos):
        pass





class ScalarField(Field):
    def __init__(self, regression_type, bounds) -> None:
        super().__init__(regression_type, bounds)
        self._regressor = None


    def fit_model(self, known_pos, known_scalars):
        self._regressor = self._regression_type()
        self._regressor.fit(known_pos, known_scalars)

    
    def predict_values(self, pos):
        return self._regressor.predict(pos)





class VectorField(Field):
    def __init__(self, regression_type, bounds) -> None:
        super().__init__(regression_type, bounds)
        self._regressors = []
        self._num_dim = None


    def fit_model(self, known_pos, known_vectors):
        self._num_dim = len(known_vectors[0])
        for i in range(self._num_dim):
            regressor = self._regression_type()
            regressor.fit(known_pos, known_vectors[:, i])
            self._regressors.append(regressor)


    def predict_values(self, pos):
        vector_out = []
        for regressor in self._regressors:
            vector_out.append(regressor.predict(pos))
        return np.array(vector_out).T