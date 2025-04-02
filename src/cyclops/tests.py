import numpy as np
import multiprocessing
from random import uniform
from pymoo.core.problem import StarmapParallelization
from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.optimize import minimize
from pyomo.environ import ConcreteModel, Var, Objective, Constraint, NonNegativeReals, minimize, SolverFactory

from cyclops.fields import Field
from cyclops.optimisers import Problem, Optimiser
from cyclops.sensor_suite import SensorSuite
from cyclops.sim_reader import MeshReader


# class Experiment:
#     """Manage the optimisers, true field and sensor suite for multi-objective optimization."""

#     def __init__(
#         self,
#         true_field: Field,
#         sensor_pos: np.ndarray[float],
#     ) -> None:
#         """Initialise class instance."""
#         self.__true_field = true_field
#         self.__num_dim = true_field.get_dim()
#         self.__sensor_pos = sensor_pos
#         self.__comparison_values = true_field.predict_values(self.__sensor_pos)

#         self.__sensor_suite = None
#         self.__repetitions = None
#         self.__problem = None

#         self.__loss_limit = None
#         self.__min_active = None
#         self.__keys = None

#     def plan_moo(
#         self,
#         sensor_suite: SensorSuite,
#         boundary_faces: np.ndarray[float],
#         repetitions=1000,
#         loss_limit=80,
#         min_active=3,
#         num_cores=8,
#     ) -> None:
#         """Prepare for Multi-Objective Optimisation (MOO)."""
#         self.__sensor_suite = sensor_suite
#         num_sensors = sensor_suite.get_num_sensors()
#         self.__problem = self.__build_problem(
#             boundary_faces, num_sensors, 2, self.calc_moo_loss, num_cores
#         )

#         self.__keys = self.__sensor_suite.calc_keys(repetitions)
#         self.__repetitions = repetitions
#         self.__loss_limit = loss_limit
#         self.__min_active = min_active

#     def __build_problem(self, boundary_faces: np.ndarray[list],
#         num_sensors: int,
#         num_obj: int,
#         loss_function: callable,
#         num_cores: int,
#     ) -> Problem:
#         """Builds the 'problem' object using Pyomo and pymoo."""
#         n_processes = num_cores
#         pool = multiprocessing.Pool(n_processes)
#         runner = StarmapParallelization(pool.starmap)

#         # Now use pymoo to handle the multi-objective optimization
#         return Problem(
#             num_dim=num_sensors * self.__num_dim,
#             num_obj=num_obj,
#             loss_function=loss_function,
#             bounds=boundary_faces,
#             elementwise_runner=runner,
#         )

#     def calc_moo_loss(self, sensor_array: np.ndarray[float]) -> list[float]:
#         """Calculate the multi-objective optimization loss."""
#         sensor_pos = sensor_array.reshape(-1, self.__num_dim)
#         losses = np.zeros(self.__repetitions)
#         for i, key in enumerate(self.__keys):
#             num_active = np.sum(key)
#             if num_active >= self.__min_active:
#                 self.__sensor_suite.set_active_sensors(key)
#                 losses[i] = self.get_MSE(sensor_pos)
#             else:
#                 losses[i] = -1
#         for i, loss in enumerate(losses):
#             if loss == -1:
#                 losses[i] = np.max(losses)

#         expected_loss = np.mean(losses)
#         failure_chance = (
#             losses > self.__loss_limit
#         ).sum() / self.__repetitions
#         return [expected_loss, failure_chance]

#     def design(self) -> any:
#         """Design the experiment using Pyomo and pymoo."""
#         return self.__optimiser.optimise(self.__problem)

#     def run_moo(self) -> None:
#         """Run Multi-Objective Optimization using pymoo."""
#         # Define a pymoo algorithm (NSGA-II or others)
#         algorithm = GA(
#             pop_size=100,
#             sampling=self.__problem.create_sampling(),
#             crossover=self.__problem.create_crossover(),
#             mutation=self.__problem.create_mutation(),
#             eliminate_duplicates=True
#         )

#         # Use pymoo to minimize the problem
#         result = minimize(self.__problem, algorithm, termination=('n_gen', 100))

#         # Output the result of the optimization
#         print("Optimized Sensor Positions:", result.F)
#         print("Pareto Front:", result.F)
#         print("Optimization History:", result.history)

