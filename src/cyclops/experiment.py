"""
Experiment class for cyclops.

Handles ground truth and sensor suite optimisation.

(c) Copyright UKAEA 2023.
"""
import numpy as np
import multiprocessing
import cyclops.sensors as sensors
import cyclops.fields as fields

from random import uniform
from pymoo.core.problem import StarmapParallelization
from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.optimize import minimize

from cyclops.fields import Field
from cyclops.optimisers import Problem, Optimiser
from cyclops.regressors import RegressionModel
from cyclops.sensor_suite import SensorSuite
from cyclops.sim_reader import MeshReader, compute_face_normal
from cyclops.pyomo_problem import SensorArrayOptimiser

class Experiment:
    """Manage the optimisers, true field and sensor suite.

    This class serves three main purposes:
    1. Initialisation to define experiment parameters.
    2. Planning to prepare for optimisation.
    3. Design to optimise the experiment.
    """
# Unsure about the optimiser input being only one. Should just assume pyomo model will be 
# used rather than making this part of the optimiser options? Use optimiser options only
# for the MOO optimiser? Adding ALL possibly needed input variables to whittle down later
    def __init__(
        self,
        reader: MeshReader,
        no_sensors: int,
        sensor_types: list,
        field_regressor: RegressionModel,
        field_positions: np.ndarray,
        optimiser: Optimiser,
        field_type: str,
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
        self._reader = reader
        self._field_type = field_type
        self._no_sensors = no_sensors
        self._sensor_types = sensor_types
        self._field_pos = field_positions

        self._regions = reader.read_region_names()
        self._point_dict = reader.point_data.keys()
        self._boundary_faces=reader.get_boundary_faces()
        self._initialised_field = self.create_field(
                                        field_type=self._field_type,
                                        regressor=field_regressor)
        self._num_dim = self._initialised_field.get_dim()
        
        sensor_pos, face_indices, face_coords = self.get_initial_sensor_pos(
                                        boundary_faces=self._boundary_faces,
                                        num_sensors=self._no_sensors)
        print("sensor_pos is ", sensor_pos)
        print("face_face_indices is ",face_indices)
        print("face_coords is ", face_coords)
        self._sensor_pos = sensor_pos
        self._face_indices = face_indices       
        self._face_coords = face_coords
        self._tri_boundary_faces = reader.get_surface_triangles()
        self._optimiser = optimiser
        self._sensor_suite = None
        self._problem = None

        #self.__loss_limit = None
        #self.__min_active = None
        #self.__keys = None

    def create_field(self, field_type, regressor):
        """
        Function to initialise the field being read from the input mesh.
        """
        # Wondering about alternative ways to set up field, currently require bounds
        # expect having so many extra points with no values present will cause problems
        # Will need to account for this or fix method somehow.
        reader = self._reader
        point_dict = list(self._point_dict)
        set_name = self._regions[:3]

       # Ensure the class exists in the sensors module
        if not hasattr(fields, field_type):
            raise ValueError(f"Unknown field type: {field_type}")
        
        # Get the class reference dynamically
        field_class = getattr(fields, field_type)

        # Ensure it's a subclass of Sensor (to prevent incorrect lookups)
        if not issubclass(field_class, fields.Field):
            raise ValueError(f"{field_class} is not a valid Field subclass.")

        new_field = field_class.mesh_reader_init(mesh_reader=reader, set_name=set_name,
                    field_2_measure=point_dict, regression_type=regressor)
        
        return new_field
    
    def create_sensor(self, sensor_type, noise, failure_fn, sensor_index, radius=None,
                      norm_vector=None):
        """
        Function to initialise a sensor.
        """
        initial_pos = self._sensor_pos[sensor_index]
        print(initial_pos)

        # Ensure the class exists in the sensors module
        if not hasattr(sensors, sensor_type):
            raise ValueError(f"Unknown sensor type: {sensor_type}")
        
        # Get the class reference dynamically
        sensor_class = getattr(sensors, sensor_type)

        # Ensure it's a subclass of Sensor (to prevent incorrect lookups)
        if not issubclass(sensor_class, sensors.Sensor):
            raise ValueError(f"{sensor_type} is not a valid Sensor subclass.")
        
        # Dynamically instantiate the correct sensor type
        if issubclass(sensor_class, sensors.PointSensor):
            new_sensor = sensor_class(centre_point=initial_pos,
                                    field_dim=self._num_dim,
                                    field=self._initialised_field,
                                    noise_dev=noise,
                                    failure_chance=failure_fn,
                                    offset_function=None)
        
        elif issubclass(sensor_class, sensors.RoundSensor):
            if norm_vector is None:
                raise ValueError("RoundSensor requires 'norm_vector'")
            
            new_sensor = sensor_class(field_dim=self._num_dim,
                                    field=self._initialised_field,
                                    centre_point=initial_pos,
                                    noise_dev=noise,
                                    failure_chance=failure_fn,
                                    norm_vector=norm_vector,
                                    offset_function=None)
        
        return new_sensor

    def plan_pyomo_problem(self, min_dist, reader, noise_list, failure_percent):
        """Function to setup a pyomo problem to optimise the position of the
        various sensors on the surface of the given mesh."""

        reader = self._reader
        tri_boundary_faces = self._tri_boundary_faces
        boundary_faces = self._boundary_faces
        sensors = self._sensor_types
        sensor_list = []

        # If failure_percent is a single value, expand it to match the length of noise_list
        noise_list = np.full(len(sensors), noise_list)
        
        # If failure_percent is a single value, expand it to match the length of noise_list
        failure_percent = np.full(len(sensors), failure_percent)

        for index, snsr in enumerate(sensors):
            face_coords = self._face_coords[index]
            print("face_coords is ", face_coords)
            noise = noise_list[index]           # Get noise for current sensor
            fail_rate = failure_percent[index]  # Get failure rate for current sensor

            # Compute normal vector if needed
            norm_vector = None
            if snsr == "RoundSensor":
                norm_vector = compute_face_normal(vertex_coords=face_coords)

            # Create sensor instance
            initialised_snsr = self.create_sensor(
                sensor_type=snsr,
                noise=noise,
                failure_fn=fail_rate,
                sensor_index=index,
                norm_vector=norm_vector  # None for PointSensor, computed for RoundSensor
            )
            
            sensor_list.append(initialised_snsr)


        new_pyomo_prob = SensorArrayOptimiser(
                                    mesh_faces = tri_boundary_faces,
                                    num_sensors = self._no_sensors,
                                    sensors = sensor_list,
                                    pos_3D = self._field_pos,
                                    true_field_values = self._initialised_field,
                                    min_distance = min_dist
                                                     )
        
        return new_pyomo_prob

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
        self._sensor_suite = sensor_suite
        num_sensors = sensor_suite.get_num_sensors()
        self._problem = self._build_problem(
            sensor_bounds, num_sensors, 1, self.calc_SOO_loss, num_cores
        )
        self._repetitions = repetitions

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
        self._sensor_suite = sensor_suite
        num_sensors = sensor_suite.get_num_sensors()
        self._problem = self._build_problem(
            boundary_faces, num_sensors, 2, self.calc_moo_loss, num_cores
        )

        self._keys = self._sensor_suite.calc_keys(repetitions)
        self._repetitions = repetitions
        self._loss_limit = loss_limit
        self._min_active = min_active

    @staticmethod
    def select_tri_point(v0, v1, v2) -> np.ndarray[float]:
        """ Selects a random point on a triangular cell face for sensor
        placement.
        
        Args:
            v0 (list): 1st coordinate defining a given cell face
            v1 (list): 2nd coordinate defining a given cell face
            v2 (list): 3rd coordinate defining a given cell face
                   
        Returns:
            point: np.ndarray[float] 
            face_coords: np.ndarray
        """
        face_coords = [v0, v1, v2]
        # Generate random barycentric coordinates (r1, r2)
        r1 = uniform(0, 1)
        r2 = uniform(0, 1)

        if r1 + r2 > 1:
            r1 = 1 - r1
            r2 = 1 - r2
        r3 = 1 - r1 - r2

        # Compute the point using barycentric coordinates
        point = r1 * np.array(v0) + r2 * np.array(v1) + r3 * np.array(v2)
        return point, face_coords

    @staticmethod
    def select_quad_point(v0, v1, v2, v3) -> np.ndarray[float]:
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
            face_coords: np.ndarray
        """
        # Decompose quad into two triangles
        triangle = np.random.choice([1, 2])

        point = []

        if triangle == 1:
            point, face_coords = Experiment.select_tri_point(v0, v1, v2)
        elif triangle == 2:
            point, face_coords = Experiment.select_tri_point(v0, v2, v3)

        return point, face_coords

    @staticmethod
    def select_hex_point(v0, v1, v2, v3, v4, v5) -> np.ndarray:
        """ Selects a random point on a hexagonal cell face for sensor placement.

        Args:
            v0, v1, v2, v3, v4, v5 (list): 6 coordinates defining a given hexagonal cell face

        Returns:
            point: np.ndarray
            face_coords: np.ndarray
        """
        # Decompose hexagon into 6 triangles by selecting the center point and 
        # forming triangles with consecutive vertices

        center = np.mean([v0, v1, v2, v3, v4, v5], axis=0)

        # Randomly choose one of the 6 triangles
        triangle_index = np.random.choice([0, 1, 2, 3, 4, 5])

        # Select the corresponding triangle vertices
        if triangle_index == 0:
            return Experiment.select_tri_point(center, v0, v1)
        elif triangle_index == 1:
            return Experiment.select_tri_point(center, v1, v2)
        elif triangle_index == 2:
            return Experiment.select_tri_point(center, v2, v3)
        elif triangle_index == 3:
            return Experiment.select_tri_point(center, v3, v4)
        elif triangle_index == 4:
            return Experiment.select_tri_point(center, v4, v5)
        elif triangle_index == 5:
            return Experiment.select_tri_point(center, v5, v0)

    def get_initial_sensor_pos(self, boundary_faces: np.ndarray[list],
                                  num_sensors: int)-> np.ndarray[float]:
        """Generate points on the surface of a given mesh for the placement of
        sensors.

        Args:
            boundary_faces (list): list of cell faces

        Returns:
            positions (np.ndarray): n-long np.ndarray of point coordinates to
            place sensors at
        """
        positions = []
        faces = []
        face_coord_vertices =[]

        for _ in range(num_sensors):
            # Randomly select a boundary face (note that all cell types can
            # only have triangular or quadrilateral faces so we only need to cover two cases)
            face_num = np.random.randint(len(boundary_faces))
            face = boundary_faces[face_num]
            vertices = MeshReader.read_points(self._reader)

            face_vertices = vertices[face]
            if len(face_vertices) == 3:  # Tri
                v0, v1, v2 = face_vertices
                point, face_coords = self.select_tri_point(face, v0, v1, v2)
                positions.append(point)
                faces.append(face_num)
                face_coord_vertices.append(face_coords)

            elif len(face_vertices) == 4:  # Quad
                v0, v1, v2, v3 = face_vertices
                point, face_coords = self.select_quad_point(v0, v1, v2, v3)
                positions.append(point)
                faces.append(face_num)
                face_coord_vertices.append(face_coords)
            
            elif len(face_vertices) == 6:  # Hexagon
                v0, v1, v2, v3, v4, v5 = face_vertices
                point, face_coords = self.select_hex_point(v0, v1, v2, v3, v4, v5)
                positions.append(point)
                faces.append(face_num)
                face_coord_vertices.append(face_coords)
            
        positions = np.array(positions)
        faces = np.array(faces)
        face_coord_vertices = np.array(face_coord_vertices)
        return positions, faces, face_coord_vertices

    def get_norm_vector(self, face_coords):
        """
        Function to get the normal vector to a given, (triangular) mesh face.
        """
        normal_vector = compute_face_normal(vertex_coords=face_coords)
        return normal_vector

    def _build_problem(self, boundary_faces: np.ndarray[list],
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
        return self._optimiser.optimise(self._problem)

    # def calc_moo_loss(self, sensor_array: np.ndarray[float]) -> list[float]:
    #     """Calculate the moo loss of a specific sensor layout.

    #     Args:
    #         sensor_array (np.ndarray[float]): unshaped sensor layout from
    #             optimiser.

    #     Returns:
    #         list[float]: loss list.
    #     """
    #     sensor_pos = sensor_array.reshape(-1, self.__num_dim)
    #     losses = np.zeros(self.__repetitions)
    #     for i, key in enumerate(self.__keys):
    #         num_active = np.sum(key)
    #         if num_active >= self.__min_active:
    #             self.__sensor_suite.set_active_sensors(key)
    #             losses[i] = self.get_MSE(sensor_pos)
    #         else:
    #             losses[i] = -1
    #     for i, loss in enumerate(losses):
    #         if loss == -1:
    #             losses[i] = np.max(losses)

    #     expected_loss = np.mean(losses)
    #     failure_chance = (
    #         losses > self.__loss_limit
    #     ).sum() / self.__repetitions
    #     return [expected_loss, failure_chance]

    # def calc_SOO_loss(self, sensor_array: np.ndarray[float]) -> list[float]:
    #     """Calculate loss for SOO.

    #     Args:
    #         sensor_array (np.ndarray[float]): unshaped sensor layout from
    #             optimiser.

    #     Returns:
    #         list[float]: loss list.
    #     """
    #     sensor_pos = sensor_array.reshape(-1, self.__num_dim)
    #     losses = np.zeros(self.__repetitions)
    #     for i in range(self.__repetitions):
    #         losses[i] = self.get_MSE(sensor_pos)
    #     return [np.mean(losses)]

    # This can potentially stay with little change
    # def get_MSE(self, sensor_pos: np.ndarray[float]) -> float:
    #     """Calculate Mean Squared Error (MSE) from an array sensor positions.

    #     1. Update the sensor suite to the values at those positions.
    #     2. See what the sensor suite predicts the rest of the field would be.
    #     3. Calculate MSE.

    #     Args:
    #         sensor_pos (np.ndarray[float]): n by d array of n sensor positions
    #             of d dimensions.

    #     Returns:
    #         float: the MSE.
    #     """
    #     self.__sensor_suite.set_sensor_pos(sensor_pos)
    #     sensor_sites = self.__sensor_suite.get_sensor_sites()
    #     site_values = self.__true_field.predict_values(sensor_sites)
    #     self.__sensor_suite.fit_sensor_model(site_values)

    #     predicted_values = self.__sensor_suite.predict_data(
    #         self.__sensor_pos
    #     )
    #     return np.mean(np.square(predicted_values - self.__comparison_values))

    # def get_SOO_plotting_arrays(
    #     self, sensor_array: np.ndarray[float]
    # ) -> tuple:
    #     """Find the necessary data to plot plots of the potential sensor setup.

    #     Args:
    #         sensor_array (np.ndarray[float]): array of unshaped sensor
    #             positions from optimiser.

    #     Returns:
    #         tuple: Contains all plotting arrays needed.
    #     """
    #     num_sensors = self.__sensor_suite.get_num_sensors()
    #     self.__sensor_suite.set_active_sensors(np.array([True] * num_sensors))
    #     sensor_pos = sensor_array.reshape(-1, self.__num_dim)
    #     self.__sensor_suite.set_sensor_pos(sensor_pos)
    #     sensor_sites = self.__sensor_suite.get_sensor_sites()
    #     site_values = self.__true_field.predict_values(sensor_sites)
    #     self.__sensor_suite.fit_sensor_model(site_values)

    #     predicted_values = self.__sensor_suite.predict_data(
    #         self.__sensor_pos
    #     )
    #     estimated_sensor_values = self.__sensor_suite.predict_data(sensor_pos)
    #     return (
    #         sensor_pos,
    #         self.__comparison_values,
    #         predicted_values,
    #         estimated_sensor_values,
    #     )

# Using PyMOO to optimize the sensor placement
# optimiser = SensorArrayOptimiser(mesh, num_sensors=10, sensor_types=[sensor_type1, sensor_type2])
# moo_problem = MOOProblem(optimiser, num_sensors=10)
# algorithm = GA(pop_size=100)
# res = minimize(moo_problem, algorithm, termination=("n_gen", 200))