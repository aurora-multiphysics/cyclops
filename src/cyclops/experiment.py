"""
Experiment class for cyclops.

Handles ground truth and sensor suite optimisation.

(c) Copyright UKAEA 2023.
"""
import numpy as np
import multiprocessing

from cyclops.fields import Field
from cyclops.optimisers import Problem, Optimiser
from cyclops.sensor_suite import SensorSuite
from cyclops.sim_reader import MeshReader
from random import choice, uniform
from pymoo.core.problem import StarmapParallelization
from shapely.geometry import Point, Polygon

class Experiment:
    """Manage the optimisers, true field and sensor suite.

    This class serves three main purposes:
    1. Initialisation to define experiment parameters.
    2. Planning to prepare for optimisation.
    3. Design to optimise the experiment.
    """

    def __init__(
        self,
        true_field: Field,
        sensor_pos: np.ndarray[float],
        optimiser: Optimiser,
    ) -> None:
        """Initialise class instance.

        Parameters:
            true_field (Field): the simulated field which acts as the ground
                truth against which to compare the predicted field.
            sensor_pos (np.ndarray[float]): the sensor positions used to
                compare the true field to the predicted field.
            optimiser (Optimiser):
                the optimiser used to optimise sensor layout.
        """
        self.__true_field = true_field
        self.__num_dim = true_field.get_dim()
        self.__sensor_pos = sensor_pos
        self.__comparison_values = true_field.predict_values(
            self.__sensor_pos
        )

        self.__optimiser = optimiser
        self.__sensor_suite = None
        self.__repetitions = None
        self.__problem = None

        self.__loss_limit = None
        self.__min_active = None
        self.__keys = None

    def plan_soo(
        self,
        sensor_suite: SensorSuite,
        sensor_bounds: np.ndarray[float],
        repetitions=10,
        num_cores=8,
    ) -> None:
        """Prepare for Single-Objective Optimisation (SOO).

        Args:
            sensor_suite (SensorSuite): the collection of sensors used for
                the experiment.
            sensor_bounds (np.ndarray[float]): bounds within which a sensor can
                be placed.
            repetitions (int, optional): number of repetitions to average
                error over. Defaults to 10.
        """
        self.__sensor_suite = sensor_suite
        num_sensors = sensor_suite.get_num_sensors()
        self.__problem = self.__build_problem(
            sensor_bounds, num_sensors, 1, self.calc_SOO_loss, num_cores
        )
        self.__repetitions = repetitions

    def plan_moo(
        self,
        sensor_suite: SensorSuite,
        boundary_faces: np.ndarray[float],
        repetitions=1000,
        loss_limit=80,
        min_active=3,
        num_cores=8,
    ) -> None:
        """Prepare for Multi-Objective Optimisation (MOO).

        Args:
            sensor_suite (SensorSuite): the collection of sensors used for
                the experiment.
            boundary_faces (np.ndarray[float]): bounds within which a sensor can
                be placed.
            repetitions (int, optional): number of repetitions to average
                error over. Defaults to 10.
            loss_limit (_type_, optional): maximum MSE for a successful
                experiment. Defaults to 80.
            min_active (int, optional): if the number of active sensors falls
            below this value due to emulated sensor failure, the loss is set to
            the maximum i.e. it is assumed the experiment would be invalidated.
            Defaults to 3.
        """
        self.__sensor_suite = sensor_suite
        num_sensors = sensor_suite.get_num_sensors()
        self.__problem = self.__build_problem(
            boundary_faces, num_sensors, 2, self.calc_moo_loss, num_cores
        )

        self.__keys = self.__sensor_suite.calc_keys(repetitions)
        self.__repetitions = repetitions
        self.__loss_limit = loss_limit
        self.__min_active = min_active

    def select_tri_point(self, v0, v1, v2) -> np.ndarray[float]:
        """ Selects a random point on a triangular cell face for sensor
        placement.
        
        Args:
            v0 (list): 1st coordinate defining a given cell face
            v1 (list): 2nd coordinate defining a given cell face
            v2 (list): 3rd coordinate defining a given cell face
                   
        Returns:
            point: np.ndarray[float] 
        """
        # Generate random barycentric coordinates (r1, r2)
        r1 = uniform(0, 1)
        r2 = uniform(0, 1)

        if r1 + r2 > 1:
            r1 = 1 - r1
            r2 = 1 - r2
        r3 = 1 - r1 - r2

        # Compute the point using barycentric coordinates
        point = r1 * np.array(v0) + r2 * np.array(v1) + r3 * np.array(v2)
        return point

    def select_quad_point(self, v0, v1, v2, v3) -> np.ndarray[float]:
        # To do - need to balance probabilities on triangle surfaces vs
        # quadrilateral by area, currently points are twice as likely to
        # generate on triangles
        """ Selects a random point on a quadrilateral cell face to place for
        sensor placement.

        Args:
            v0 (list): 1st coordinate defining a given cell face
            v1 (list): 2nd coordinate defining a given cell face
            v2 (list): 3rd coordinate defining a given cell face
            v3 (list): 4th coordinate defining a given cell face
                   
        Returns:
            point: np.ndarray[float] 
        """
        # Decompose quad into two triangles
        triangle = np.random.choice([1, 2])

        if triangle == 1:
            point = Experiment.select_tri_point(v0, v1, v2)
        elif triangle == 2:
            point = Experiment.select_tri_point(v0, v2, v3)    

        return point

    def generate_sensor_positions(self, boundary_faces: np.ndarray[list],
                                  num_sensors: int)-> np.ndarray[float]:
        """Generate points on the surface of a given mesh for the placement of
        sensors.

        Args:
            boundary_faces (list): list of cell faces

        Returns:
            positions (np.ndarray): n long numpy array of point coordinates to
            place sensors at
        """
        positions = []

        for _ in range(num_sensors):
            # Randomly select a boundary face (note that all cell types can
            # only have triangular or quadrilateral faces so we only need to cover two cases)
            face = boundary_faces[np.random.randint(len(boundary_faces))]
            vertices = MeshReader.read_points(self)
            face_vertices = vertices[face]

            if len(face) == 3:  # Tri
                v0, v1, v2 = map(lambda idx: face_vertices[idx], [0, 1, 2])
                point = Experiment.select_tri_point(face, v0, v1, v2)

            elif len(face) == 4:  # Quad
                v0, v1, v2, v3 = map(lambda idx: face_vertices[idx], [0, 1, 2, 3])
                point = Experiment.select_quad_point(face, v0, v1, v2, v3)
            
            positions.append(point)

        positions = np.array(positions)
        return positions

    def __build_problem(self, boundary_faces: np.ndarray[list],
        num_sensors: int,
        num_obj: int,
        loss_function: callable,
        num_cores: int,
    ) -> Problem:
        """Builds the 'problem' object, which contains the sensors, their
        positions, the no. of objectives and the loss function to be
        minimised.

        Args:
            boundary_faces (np.ndarray[float]): list of faces that a sensor
            can be placed on.
            num_sensors (int): number of sensors to use.
            num_obj (int): number of objectives to optimiser for.
            loss_function (callable): loss function to minimise.

        Returns:
            Problem: problem object to optimise.
        """
        n_processes = num_cores
        pool = multiprocessing.Pool(n_processes)
        runner = StarmapParallelization(pool.starmap)

        return Problem(
            num_dim=num_sensors * self.__num_dim,
            num_obj=num_obj,
            loss_function=loss_function,
            bounds=boundary_faces,
            elementwise_runner=runner,
        )

    def design(self) -> any:
        """Design experiment.

        Returns:
            any: results object containing Pareto-optimal layouts and
                optimisation history.
        """
        return self.__optimiser.optimise(self.__problem)

    def calc_moo_loss(self, sensor_array: np.ndarray[float]) -> list[float]:
        """Calculate the moo loss of a specific sensor layout.

        Args:
            sensor_array (np.ndarray[float]): unshaped sensor layout from
                optimiser.

        Returns:
            list[float]: loss list.
        """
        sensor_pos = sensor_array.reshape(-1, self.__num_dim)
        losses = np.zeros(self.__repetitions)
        for i, key in enumerate(self.__keys):
            num_active = np.sum(key)
            if num_active >= self.__min_active:
                self.__sensor_suite.set_active_sensors(key)
                losses[i] = self.get_MSE(sensor_pos)
            else:
                losses[i] = -1
        for i, loss in enumerate(losses):
            if loss == -1:
                losses[i] = np.max(losses)

        expected_loss = np.mean(losses)
        failure_chance = (
            losses > self.__loss_limit
        ).sum() / self.__repetitions
        return [expected_loss, failure_chance]

    def calc_SOO_loss(self, sensor_array: np.ndarray[float]) -> list[float]:
        """Calculate loss for SOO.

        Args:
            sensor_array (np.ndarray[float]): unshaped sensor layout from
                optimiser.

        Returns:
            list[float]: loss list.
        """
        sensor_pos = sensor_array.reshape(-1, self.__num_dim)
        losses = np.zeros(self.__repetitions)
        for i in range(self.__repetitions):
            losses[i] = self.get_MSE(sensor_pos)
        return [np.mean(losses)]

    def get_MSE(self, sensor_pos: np.ndarray[float]) -> float:
        """Calculate Mean Squared Error (MSE) from an array sensor positions.

        1. Update the sensor suite to the values at those positions.
        2. See what the sensor suite predicts the rest of the field would be.
        3. Calculate MSE.

        Args:
            sensor_pos (np.ndarray[float]): n by d array of n sensor positions
                of d dimensions.

        Returns:
            float: the MSE.
        """
        self.__sensor_suite.set_sensor_pos(sensor_pos)
        sensor_sites = self.__sensor_suite.get_sensor_sites()
        site_values = self.__true_field.predict_values(sensor_sites)
        self.__sensor_suite.fit_sensor_model(site_values)

        predicted_values = self.__sensor_suite.predict_data(
            self.__sensor_pos
        )
        return np.mean(np.square(predicted_values - self.__comparison_values))

    def get_SOO_plotting_arrays(
        self, sensor_array: np.ndarray[float]
    ) -> tuple:
        """Find the necessary data to plot plots of the potential sensor setup.

        Args:
            sensor_array (np.ndarray[float]): array of unshaped sensor
                positions from optimiser.

        Returns:
            tuple: Contains all plotting arrays needed.
        """
        num_sensors = self.__sensor_suite.get_num_sensors()
        self.__sensor_suite.set_active_sensors(np.array([True] * num_sensors))
        sensor_pos = sensor_array.reshape(-1, self.__num_dim)
        self.__sensor_suite.set_sensor_pos(sensor_pos)
        sensor_sites = self.__sensor_suite.get_sensor_sites()
        site_values = self.__true_field.predict_values(sensor_sites)
        self.__sensor_suite.fit_sensor_model(site_values)

        predicted_values = self.__sensor_suite.predict_data(
            self.__sensor_pos
        )
        estimated_sensor_values = self.__sensor_suite.predict_data(sensor_pos)
        return (
            sensor_pos,
            self.__comparison_values,
            predicted_values,
            estimated_sensor_values,
        )
