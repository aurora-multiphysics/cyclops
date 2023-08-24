from pymoo.core.problem import ElementwiseProblem




class SensorPlacementProblem(ElementwiseProblem):
    def __init__(self, num_dim, loss_function, borders):
        super().__init__(
            n_var = num_dim, 
            n_obj = 2, 
            xl = borders[0], 
            xu = borders[1]
        )
        self.__loss_function = loss_function


    def _evaluate(self, optim_array, out, *args, **kwargs):
        # Convert optim_array to sensor positions
        sensor_pos = None
        loss, deviation = self.__loss_function(sensor_pos)
        out['F'] = [loss, deviation]

