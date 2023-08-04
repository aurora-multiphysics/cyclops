from pymoo.algorithms.soo.nonconvex.pso import PSO
from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.termination import get_termination
from pymoo.core.problem import Problem
from pymoo.optimize import minimize
import numpy as np




class LossFunction(Problem):
    def __init__(self, num_sensors, model_manager):
        if model_manager.is_symmetric():
            low_border = [0, -0.0135] * (num_sensors//2)
            high_border = [0.0135, 0.0215] * (num_sensors//2)
            num_dimensions = num_sensors
        else:
            low_border = [-0.0135, 0.0135] * num_sensors
            high_border = [0.0135, 0.0215] * num_sensors
            num_dimensions = 2*num_sensors

        super().__init__(
            n_var=num_dimensions, 
            n_obj=1, 
            n_ieq_constr=0, 
            xl=low_border, 
            xu=high_border
        )
        self.__model_manager = model_manager


    def _evaluate(self, swarm_values, out, *args, **kwargs):
        out['F'] = np.apply_along_axis(self.__model_manager.get_loss, 1, swarm_values)




def optimise_with_GA(problem, time_limit):
    algorithm = GA(
        pop_size=50,
        eliminate_duplicates=True)
    termination = get_termination("time", time_limit)

    res = minimize(problem,
                algorithm,
                termination,
                seed=3,
                save_history=True,
                verbose=True)
    return res



def optimise_with_PSO(problem, time_limit):
    algorithm = PSO(
        pop_size=20,
        adaptive = True
    )
    termination = get_termination("time", time_limit)

    res = minimize(problem,
                algorithm,
                termination,
                seed=1,
                save_history=True,
                verbose=True)
    return res



