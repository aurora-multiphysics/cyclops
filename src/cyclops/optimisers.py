"""
Optimiser classes for cyclops. These handle optimisation of sensor placement.

(c) Copyright UKAEA 2023.
"""
import numpy as np

from pymoo.algorithms.soo.nonconvex.pso import PSO
from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.problem import ElementwiseProblem
from pymoo.termination import get_termination
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.optimize import minimize


class Problem(ElementwiseProblem):
    """Problem class; allows a function to be minimised."""

    def __init__(
        self,
        num_dim: int,
        num_obj: int,
        loss_function: callable,
        bounds: np.ndarray,
        **kwargs,
    ) -> None:
        """Set up the problem.

        Args:
            num_dim (int): number of dimensions of the function input.
            num_obj (int): number of objectives (return values) of the
                function.
            loss_function (callable): function to minimise.
            bounds (np.ndarray): the upper and lower values of the function
                domain.
        """
        super().__init__(
            n_var=num_dim, n_obj=num_obj, xl=bounds[0], xu=bounds[1], **kwargs
        )
        self.__loss_function = loss_function

    def _evaluate(
        self, optim_array: np.ndarray[float], out: dict, *args: any, **kwargs: any
    ) -> None:
        """Evaluate the loss function.

        Args:
            optim_array (np.ndarray[float]): the proposed input to the loss
                function.
            out (dict): a pymoo dictionary to store constrains and results.
        """
        out["F"] = self.__loss_function(optim_array)


class Optimiser:
    """Optimiser base class."""

    def __init__(self, time_limit: str, algorithm: any) -> None:
        """Set up the optimisers.

        Args:
            time_limit (str): the maximum time the optimiser should run for.
            algorithm (any): the kind of optimiser.
        """
        # Remove time_limit and replace with convergence conditions
        self._limit = get_termination("time", time_limit)
        self._algorithm = algorithm

    def optimise(self, problem: Problem) -> any:
        """Minimise the problem.

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
    """Multi-objective optimiser.

    Simulates evolution.
    """

    def __init__(self, time_limit) -> None:
        """Initialise class instance.

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
    """Single-objective optimiser.

    Simulates birds searching for food.
    """

    def __init__(self, time_limit) -> None:
        """Initialise class instance.

        Args:
            time_limit (_type_): time the optimisation should run for.
        """
        algorithm = PSO(pop_size=30, adaptive=True)
        super().__init__(time_limit, algorithm)


class GAOptimiser(Optimiser):
    """Single-objective optimiser.

    Simulates evolution.
    """

    def __init__(self, time_limit) -> None:
        """Initialise class instance.

        Args:
            time_limit (_type_): time the optimisation should run for.
        """
        algorithm = GA(pop_size=50, eliminate_duplicates=True)
        super().__init__(time_limit, algorithm)
