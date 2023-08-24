from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.algorithms.soo.nonconvex.pso import PSO
from pymoo.core.problem import ElementwiseProblem
from pymoo.algorithms.moo.moead import MOEAD
from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.termination import get_termination
from pymoo.operators.crossover.sbx import SBX
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.operators.mutation.pm import PM
from pymoo.optimize import minimize
import numpy as np





class Problem(ElementwiseProblem):
    def __init__(self, num_dim, num_obj, loss_function, borders):
        # Note that loss_function must return a list of the MOO things
        super().__init__(
            n_var = num_dim, 
            n_obj = num_obj, 
            xl = borders[0], 
            xu = borders[1]
        )
        self.__loss_function = loss_function


    def _evaluate(self, optim_array, out, *args, **kwargs):
        out['F'] = self.__loss_function(optim_array)




class Optimiser():
    def __init__(self, time_limit, algorithm) -> None:
        self.__limit = get_termination("time", time_limit)
        self.__algorithm = algorithm


    def optimise(self, problem):
        res = minimize(
            problem,
            self.__algorithm,
            self.__limit,
            seed = 1,
            save_history = True,
            verbose = True
        )
        return res





class NSGA2Optimiser(Optimiser):
    def __init__(self, time_limit) -> None:
        algorithm = NSGA2(
            pop_size=40,
            n_offsprings=10,
            sampling=FloatRandomSampling(),
            crossover=SBX(prob=0.9, eta=15),
            mutation=PM(eta=20),
            eliminate_duplicates=True
        )
        super().__init__(time_limit, algorithm)





class PSOOptimiser(Optimiser):
    def __init__(self, time_limit) -> None:
        algorithm = PSO(
            pop_size=30,
            adaptive=True
        )
        super().__init__(time_limit, algorithm)




class GAOptimiser(Optimiser):
    def __init__(self, time_limit) -> None:
        algorithm = GA(
            pop_size=50,
            eliminate_duplicates=True
        )
        super().__init__(time_limit, algorithm)