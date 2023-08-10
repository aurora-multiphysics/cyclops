from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.core.problem import ElementwiseProblem
from pymoo.termination import get_termination
from pymoo.operators.crossover.sbx import SBX
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.operators.mutation.pm import PM
from pymoo.optimize import minimize



#Monoblock values
X_BOUNDS = (-0.0135, 0.0135)
Y_BOUNDS = (-0.0135, 0.0215)
Z_BOUNDS = (0, 0.012)




class LossFunction(ElementwiseProblem):
    def __init__(self, num_sensors, model_manager):
        if model_manager.is_symmetric():
            low_border = [0, -0.0115] * (num_sensors//2)
            high_border = [0.011, 0.019] * (num_sensors//2)
            num_dimensions = num_sensors
        else:
            low_border = [-0.011, -0.0115] * num_sensors
            high_border = [0.011, 0.019] * num_sensors
            num_dimensions = 2*num_sensors

        super().__init__(
            n_var=num_dimensions, 
            n_obj=2, 
            xl=low_border, 
            xu=high_border
        )
        self.__model_manager = model_manager


    def _evaluate(self, sensor_layout, out, *args, **kwargs):
        loss, deviation = self.__model_manager.get_loss(sensor_layout)
        out['F'] = [loss, deviation]




def optimise_with_GA(problem, time_limit):
    algorithm = NSGA2(
        pop_size=40,
        n_offsprings=10,
        sampling=FloatRandomSampling(),
        crossover=SBX(prob=0.9, eta=15),
        mutation=PM(eta=20),
        eliminate_duplicates=True
    )
    termination = get_termination("time", time_limit)
    
    res = minimize(problem,
                algorithm,
                termination,
                seed=2,
                save_history=True,
                verbose=True)
    return res