"""
Optimiser classes for cyclops. These handle optimisation of sensor placement.

(c) Copyright UKAEA 2023.
"""
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.algorithms.soo.nonconvex.pso import PSO
from pymoo.core.problem import ElementwiseProblem
from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.termination import get_termination
from pymoo.operators.crossover.sbx import SBX
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.operators.mutation.pm import PM
from pymoo.optimize import minimize
import numpy as np


class Problem(ElementwiseProblem):
    """
    Problem class allows a function to be minimised.
    """

    def __init__(
        self,
        num_dim: int,
        num_obj: int,
        loss_function: callable,
        borders: np.ndarray,
        **kwargs
    ) -> None:
        """
        Setup the problem.

        Args:
            num_dim (int): number of dimensions of the function input.
            num_obj (int): number of objectives (return values) of the
                function.
            loss_function (callable): function to minimise.
            borders (np.ndarray): the upper and lower values of the function
                domain.
        """
        super().__init__(
            n_var=num_dim,
            n_obj=num_obj,
            xl=borders[0],
            xu=borders[1],
            **kwargs
        )
        self.__loss_function = loss_function

    def _evaluate(
        self,
        optim_array: np.ndarray[float],
        out: dict,
        *args: any,
        **kwargs: any
    ) -> None:
        """
        Evaluates the loss function.

        Args:
            optim_array (np.ndarray[float]): the proposed input to the loss
                function.
            out (dict): a pymoo dictionary to store constrains and results.
        """
        out["F"] = self.__loss_function(optim_array)


class Optimiser:
    """
    This is an abstract class to define various kinds of optimisers from.
    """

    def __init__(self, time_limit: str, algorithm: any) -> None:
        """
        Sets up the optimisers.

        Args:
            time_limit (str): the maximum time the optimiser should run for.
            algorithm (any): the kind of optimiser.
        """
        self._limit = get_termination("time", time_limit)
        self._algorithm = algorithm

    def optimise(self, problem: Problem) -> any:
        """
        Minimises the problem.

        Args:
            problem (Problem): the problem describing the function to minimise.

        Returns:
            any: a result object containing the results and optimisation
                history.
        """
        res = minimize(
            problem,
            self._algorithm,
            self._limit,
            seed=1,
            save_history=True,
            verbose=True,
        )
        return res


class NSGA2Optimiser(Optimiser):
    """
    MOO optimiser.
    Simulates evolution.
    """

    def __init__(self, time_limit) -> None:
        """
        Args:
            time_limit (_type_): time the optimisation should run for.
        """
        algorithm = NSGA2(
            pop_size=40,
            n_offsprings=10,
            sampling=FloatRandomSampling(),
            crossover=SBX(prob=0.9, eta=15),
            mutation=PM(eta=20),
            eliminate_duplicates=True,
        )
        super().__init__(time_limit, algorithm)


class PSOOptimiser(Optimiser):
    """
    SOO optimiser.
    Simulates birds searching for food.
    """

    def __init__(self, time_limit) -> None:
        """
        Args:
            time_limit (_type_): time the optimisation should run for.
        """
        algorithm = PSO(pop_size=30, adaptive=True)
        super().__init__(time_limit, algorithm)


class GAOptimiser(Optimiser):
    """
    SOO optimiser.
    Simulates evolution.
    """

    def __init__(self, time_limit) -> None:
        """
        Args:
            time_limit (_type_): time the optimisation should run for.
        """
        algorithm = GA(pop_size=50, eliminate_duplicates=True)
        super().__init__(time_limit, algorithm)
