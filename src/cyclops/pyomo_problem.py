import numpy as np
import cyclops.sensors as Sensor
import cyclops.sensor_suite as SensorSuite
import random
import math
import pynumero
from pyomo.environ import sqrt, summation, exp
from cyclops.sim_reader import MeshReader
from random import shuffle
from sklearn.gaussian_process.kernels import RBF
from typing import List


# ToDo allow for no. of sensors to vary, should take the form of an
# additional constraint, something like:
# model.max_sensors = Constraint(expr=sum(
# model.num_sensors[i] for i in range(1, 6)) <= n)  # Max of n sensors allowed
# model.min_sensors = Constraint(expr=sum(
# model.num_sensors[i] for i in range(1, 6)) >= n)  # Min of m sensors allowed

#Need to make sure that the list of faces that is passed are all triangles...any quads or hexes need to be subdivided first!
class SensorArrayOptimiser:
    def __init__(self, mesh_faces: List, num_sensors, sensors, comparison_points,
                 pos_vals: np.ndarray, true_values, min_distance=0.5,
                 regression_type='RBF', rbf_centers=None, 
                 rbf_weights=None, gamma=None, gp_kernel=None,
                 noise_level=None, epsilon=None):
        """
        Class to set up and handle a pyomo model of sensors positioned on the
        surface of a mesh.

        Args:
        mesh_faces (List): List of (x, y, z) mesh surface coordinates, to be
                        passed after mesh analysis. Each entry corresponds to
                        a triangular face's vertices.
        num_sensors (int): Integer (max) number of sensors to be place
        sensors (array): Array containing the initial faces each sensor should
                        be placed on.
        pos_vals (np.ndarray): Numpy array of 3D positions in the true field
        true_values (np.ndarray): selected comparison points for error
                        caculation, these should correspond to the positions
                         inside "pos_vals" and come from an already interpolated
                         field.
        min_distance (float): Minimum required distance between sensors, this
                        MUST be in the same units as the size of sensors and
                        as the Field's length.
        regression_type (str): The regression model to use
        rbf_centers, rbf_weights, gamma: RBF parameters
        gp_kernel (kernel object): Kernel for Gaussian Process regression
        noise_level (float): Noise level for Gaussian Process

        Returns:
        """

        self.true_field_vals = true_values
        self.pos_vals = pos_vals
        self.epsilon = epsilon

        self.__mesh_faces = mesh_faces
        self.__face_vertex_map = {
            i: mesh_faces[i] for i in range(len(mesh_faces))
        }
        self.num_sensors = num_sensors
        self.min_distance = min_distance
        self.sensors = sensors # Used to assign initial sensor faces
        self.true_values = true_values
        self._3D_pos = pos_vals  # Initial sensor positions
        self.comparison_pts = comparison_points

        self.rbf_centers = rbf_centers        # List of 3D points
        self.rbf_weights = rbf_weights        # List of floats
        self.gamma = gamma            # Float
        self.regression_type = regression_type
        self.noise_level = noise_level
        self.gp_kernel = gp_kernel

        # Initialise the Pyomo model
        self.model = ConcreteModel()

        # Define the Sensors_pos range set first
        self.model.Sensors_pos = RangeSet(0, self.num_sensors - 1)
        
        # We define the initial sensor positions as decision variables
        # These variables will be optimised, but start from a specific mesh
        # face (so sensor_x[i], sensor_y[i], and sensor_z[i] are all
        # optimisation variables)
        self.model.sensor_x = Var(self.model.Sensors_pos, domain=Reals)
        self.model.sensor_y = Var(self.model.Sensors_pos, domain=Reals)
        self.model.sensor_z = Var(self.model.Sensors_pos, domain=Reals)

         # Face Assignment for each sensor, note this is also a decision variable
        self.model.FaceAssignment = Var(self.model.Sensors_pos, 
            within=NonNegativeIntegers, bounds=(0, len(self.__mesh_faces) - 1))

        # Initialise the decision variables (sensor positions) to centroids of
        #  triangles or based on provided sensors
        for i in range(self.num_sensors):
            # Randomly select a triangle (mesh face)
            face_index = self.sensors[i]  # This should be the index of the face the sensor is placed on
            triangle = self.__mesh_faces[face_index]
            self.model.FaceAssignment[i] = face_index
            v1, v2, v3 = triangle 
        
            # Generate random barycentric coordinates within the triangle
            random_coords = sorted([random.random() for _ in range(3)])

            # Normalise to ensure the sum of the coordinates equals 1
            total = sum(random_coords)
            barycentric_coords = [coord / total for coord in random_coords]

            # Calculate the sensor position using the barycentric coordinates
            centroid = [
            sum(barycentric_coords[j] * triangle[j][0] for j in [0, 1, 2]),
            sum(barycentric_coords[j] * triangle[j][1] for j in [0, 1, 2]),
            sum(barycentric_coords[j] * triangle[j][2] for j in [0, 1, 2]) 
            ]

            # Set initial sensor positions at the calculated point, recall
            # that i is indexing the list of sensors, so if i=3 then
            # sensor_x[i] is the x coord of the 3rd sensor
            self.model.sensor_x[i] = centroid[0]
            self.model.sensor_y[i] = centroid[1]
            self.model.sensor_z[i] = centroid[2]

        # Define vertex coords as a Pyomo Param
        self.model.FaceVertices = Param(
            self.model.Sensors_pos, [1, 2, 3],
            initialize=lambda model, i, j: self.__face_vertex_map[i][j - 1][0] if j == 1 else (
                self.__face_vertex_map[i][j - 1][1] if j == 2 else self.__face_vertex_map[i][j - 1][2]
            ),
            within=Reals
        )

        # Define BaryCoords as decision variables for each sensor AND each of
        # the 3 barycentric coordinates (weights for tri vertices)
        self.model.BaryCoords = Var(self.model.Sensors_pos,
                                    [1, 2, 3], within=UnitInterval)

        # Add Barycentric sum constraint
        self.model.BaryConstraint = Constraint(self.model.Sensors_pos,
                                        rule=self.barycentric_sum_rule)

        # Define sensor positions as expressions based on barycentric coordinates
        self.model.sensor_x_expr = Expression(self.model.Sensors_pos,
                                             rule=self.sensor_position_x)
        self.model.sensor_y_expr = Expression(self.model.Sensors_pos,
                                             rule=self.sensor_position_y)
        self.model.sensor_z_expr = Expression(self.model.Sensors_pos,
                                             rule=self.sensor_position_z)
                        
        # Constraint: Sensors should be at least min_distance apart
        self.model.sensor_dist_constraints = Constraint(range(
            self.num_sensors), range(self.num_sensors),
            rule=self._sensor_distance_constraint)

        # Call the build model method, passing the model
        self._build_model(self.model)

        # Objective functions
        #self.model.obj = Objective(rule=self.reconstruction_error, sense=minimize)

    def barycentric_sum_rule(self, model, i):
        """Ensure that barycentric coordinates sum to 1
        
        Args:
            model (pyomo model): the current pyomo model instance
            i (int): the index corresponding to the coordinates to be checked
            (i.e. coordinates of a sensor)

        Returns:
            the barycentric sum rule
        """
        return model.BaryCoords[i, 1] + model.BaryCoords[i, 2] + model.BaryCoords[i, 3] == 1

    def _build_model(self, model):
        """Build model based on the selected regression type."""
        if self.regression_type == 'RBF':
            self._build_rbf_model(model)
        elif self.regression_type == 'GaussianProcess':
            self._build_gp_model(model)
        else:
            raise ValueError(f"Unsupported regression type: {self.regression_type}")

    def _build_gp_model(self, model):
        """
        Build the model using Gaussian Process regression.
        """

        # # Define Sets
        # model.D = RangeSet(3)  # Dimensionality of the field (3D in this case)
        # model.CenterPt_Indices = RangeSet(self.num_sensors)
        
        # model.NoiseLevel = Param(initialize=self.noise_level)

        # # We need to define the kernel (covariance) matrix for sensor positions
        # # This is K(x, x') = kernel(x, x')
        # model.Kernel = Param(model.CenterPt_Indices, model.CenterPt_Indices, initialize=self.kernel_function_rule)

        # # Use Gaussian Process model to make predictions, set up the expressions
        # # for field predictions at sensor positions. This requires integrating 
        # # the GP predictions into the model.
        # model.CovarianceMatrix = Param(
        #     model.CenterPt_Indices, model.CenterPt_Indices, initialize=lambda m, i, j: self.kernel_function(
        #         m.sensor_x[i], m.sensor_y[i], m.sensor_z[i],
        #         m.sensor_x[j], m.sensor_y[j], m.sensor_z[j] ),
        #     within=NonNegativeReals
        # )

        # # Compute the predicted values based on the GP model
        # model.PredictedValue = Expression(
        #     model.CenterPt_Indices, rule=self.gp_expr_rule
        # )

        # # Parameterises the "true values" at the same positions
        # model.TrueValue = Param(model.Sensors_pos, initialize={
        #     i+1: val for i, val in enumerate(self.true_values)
        # })
        
        # # Objective function (minimise the MSE)
        # model.MSE = Objective(
        #     expr=sum(
        #         (model.PredictedValue[i] - model.TrueValue[i])**2 
        #         for i in model.CenterPt_Indices
        #     ), sense=minimize
        # )

    def _build_rbf_model(self, model):
        """
        Build the model using RBF regression.
        """

        # Define Sets
        self.model.D = RangeSet(3)  # Dimensionality of the field (3D in this case)
        self.model.CenterPt_Indices = RangeSet(self.num_sensors)
        self.model.NoiseLevel = Param(initialize=self.noise_level)

        # We need to define the kernel (covariance) matrix for sensor positions
        # This is K(x, x') = kernel(x, x')
        self.model.Kernel = Param(self.model.CenterPt_Indices, self.model.CenterPt_Indices, initialize=self.kernel_function_rule)

        # Parameterises the "true values" at the same positions
        self.model.TrueValue = Param(self.model.Sensors_pos, initialize={
            i+1: val for i, val in enumerate(self.true_values)
        })

        # Create the regression field from sample points
        recon_gamma, recon_weights, recon_rbf_block, recon_points = RBFBlockSample(
            self.sample_points,
            self.model.Sensors_pos)
        
        self.recon_gamma = recon_gamma
        self.recon_weights = recon_weights
        self.recon_rbf_block = recon_rbf_block
        self.recon_points = recon_points

        # Define the sensor positions' predicted values (reconstructed field values)
        self.model.predicted_values = Param(self.model.Sensors_pos, initialize={
            i: self.predict_field_value(self._3D_pos[i]) for i in range(self.num_sensors)
        })

        self.model.obj = Objective(rule=obj_rule, sense=minimize)

        # Define objective: minimise the difference between truth and RBF
        # predictions at the sensor positions
        def obj_rule(model):
            prediction_diff = 0
            m = len(self.comparison_pts)
            for i in range(m):
                # Predict field value using the RBF blocks
                # for the sensor position
                fixed_prediction = model.RBF_fixed.predict_new_point(
                    [self.comparison_pts[i, 0],
                    self.comparison_pts[i, 1],
                    self.comparison_pts[i, 2]])
                
                optimised_prediction = model.RBF_optimised.predict_new_point(
                    [self.comparison_pts[i, 0],
                    self.comparison_pts[i, 1],
                    self.comparison_pts[i, 2]])
                
                prediction_diff += (fixed_prediction - optimised_prediction)**2
            return prediction_diff

    def rbf_kernel_expr(self, i, j):
        """
        Computes the RBF (Gaussian) kernel between two points i and j.
        """
        x1, y1, z1 = self.rbf_centers[i, 0], self.rbf_centers[i, 1], self.rbf_centers[i, 2]
        x2, y2, z2 = self.rbf_centers[j, 0], self.rbf_centers[j, 1], self.rbf_centers[j, 2]
        distance_sq = (x1 - x2)**2 + (y1 - y2)**2 + (z1 - z2)**2
        return exp(-self.epsilon * distance_sq)

    def kernel_function_rule(self, m, i, j):
        """
        Rule to compute the covariance between sensor positions, uses RBF kernel function.
        """
        x1, y1, z1 = m.sensor_x[i], m.sensor_y[i], m.sensor_z[i]
        x2, y2, z2 = m.sensor_x[j], m.sensor_y[j], m.sensor_z[j]
        distance_sq = (x1 - x2)**2 + (y1 - y2)**2 + (z1 - z2)**2
        
        return exp(-self.gamma * distance_sq)

    # def compute_covariance_matrix(self):
    #     """
    #     Compute the covariance matrix using the kernel function.
    #     """
    #     num_sensors = len(self._3D_pos)
    #     covariance_matrix = np.zeros((num_sensors, num_sensors))

    #     # Calculate the covariance between all pairs of sensors
    #     for i in range(num_sensors):
    #         for j in range(num_sensors):
    #             # Calculate patial covariance using the kernel function
    #             # (RBF, for example)
    #             spatial_cov = self.kernel_function(self._3D_pos[i][0], 
    #                                                self._3D_pos[i][1],
    #                                                self._3D_pos[i][2], 
    #                                                self._3D_pos[j][0],
    #                                                self._3D_pos[j][1], 
    #                                                self._3D_pos[j][2])

    #         # Get the predicted (reconstructed) field values for sensors i and j using the regressor
    #         reconstructed_field_i = self.predict_field_value(self._3D_pos[i])
    #         reconstructed_field_j = self.predict_field_value(self._3D_pos[j])

    #         # Calculate the covariance between the reconstructed field values at sensors i and j
    #         field_cov = reconstructed_field_i * reconstructed_field_j

    #         # Combine the spatial and field value covariance
    #         covariance_matrix[i, j] = spatial_cov * field_cov
        
    #     return covariance_matrix

    def to_cartesian(self, model, i):
        """Convert barycentric coordinates back to cartesian coordinates
        
        Args:
            model (pyomo model): the current pyomo model instance
            i (int): the index corresponding to the coordinates to be
            converted (i.e. coordinates of a sensor)

        Returns:
            3D sensor cartesian sensor coordinates corresponding to input
            barycentric coordinates"""
        # sensors holds the indices of the chosen faces for all
        # sensors, hence select a single entry which corresponds to the face
        # index for a single sensor
        face_index = self.sensors[i]
        v1, v2, v3 = self.__mesh_faces[face_index]
        b1 = model.BaryCoords[i, 1]
        b2 = model.BaryCoords[i, 2]
        b3 = model.BaryCoords[i, 3]

        # v1, v2, and v3 are each 3D coordinates like (x, y, z)
        return [v1[0] * b1 + v2[0] * b2 + v3[0] * b3,
                v1[1] * b1 + v2[1] * b2 + v3[1] * b3,
                v1[2] * b1 + v2[2] * b2 + v3[2] * b3]

    def _sensor_distance_constraint(self, model, i, j):
        """Ensure sensors are at least `min_distance` apart.
        
        Args:
            model (pyomo model): the current pyomo model instance
            sensor_idx_i (int): the index of a sensor for distance checking
            sensor_idx_j (int): the index of a sensor for distance checking
        Returns:
            Boolean indicating whether or not distance constraint has been met
        """
        if i >= j:
        # if the same sensor is compared to itself we skip the constraint
            return Constraint.Skip

        # pi & pj are positions of sensors i & j
        pi = self.to_cartesian(model, i)
        pj = self.to_cartesian(model, j)

        # k in range(3) means this is covering X,Y,Z
        dist_sq = sum((pi[k] - pj[k]) ** 2 for k in range(3))
        return dist_sq >= self.min_distance ** 2 

    def sensor_position_x(self, model, i):
        """
        Expression for the x-coordinate of the sensor position based on
        barycentric coordinates
        """
        face_index = int(model.FaceAssignment[i].value)

        v1, v2, v3 = self.__mesh_faces[face_index]

        b1 = model.BaryCoords[i, 1]
        b2 = model.BaryCoords[i, 2]
        b3 = model.BaryCoords[i, 3]
        # Calculate the x-coordinate from barycentric coordinates
        return v1[0] * b1 + v2[0] * b2 + v3[0] * b3

    def sensor_position_y(self, model, i):
        """
        Expression for the y-coordinate of the sensor position based on
        barycentric coordinates
        """
        face_index = int(model.FaceAssignment[i].value)

        v1, v2, v3 = self.__mesh_faces[face_index]

        b1 = model.BaryCoords[i, 1]
        b2 = model.BaryCoords[i, 2]
        b3 = model.BaryCoords[i, 3]
        # Calculate the y-coordinate from barycentric coordinates
        return v1[1] * b1 + v2[1] * b2 + v3[1] * b3

    def sensor_position_z(self, model, i):
        """
        Expression for the z-coordinate of the sensor position based on
        barycentric coordinates
        """
        face_index = int(model.FaceAssignment[i].value)
        v1, v2, v3 = self.__mesh_faces[face_index]
        b1 = model.BaryCoords[i, 1]
        b2 = model.BaryCoords[i, 2]
        b3 = model.BaryCoords[i, 3]
        # Calculate the z-coordinate from barycentric coordinates
        return v1[2] * b1 + v2[2] * b2 + v3[2] * b3

    # def reconstruction_error(self, model):
    #     """Define the objective function to minimise the field reconstruction
    #     error.
        
    #     Args:
    #         true_values (cyclops field): the 'ground truth' field being sampled
    #         and compared against
    #         sensors (list of cyclops sensors): a list of the sensor types that
    #         will be used in the simulation, these types should be initialised
    #         instances of those sensors.
    #     Returns:
    #         self.model.obj (pyomo objective): this is the objective function
    #         that will be solved by pyomo to obtain an answer to the overall
    #         optimisation problem.
    #     """
    #     sensor_pos = [value(self.to_cartesian(model, i)) for i in model.Sensors_pos]
    #     sensor_pos = np.array(sensor_pos)

    #     return sum((model.PredictedValue[i] - model.TrueValue[i])**2 for i in
    #                model.Sensors_pos)

    # def update_sensor_positions(self):
    #     """Update optimised sensor positions from the Pyomo model after solving."""
    #     optimised_positions = [self.to_cartesian(self.model, i) for i in
    #                                 self.model.Sensors_pos]
    #     return optimised_positions

    # def evaluate(self, sensor_positions):
    #     """Update model, run optimisation problem, and return the objectives."""
    #     self.update_sensor_positions(sensor_positions)
    #     self.solve()
        
    #     # objectives (MSE, failure probability, etc.)
    #     obj_values = [self.model.obj()]
    #     return np.array(obj_values)   

    def solve(self):
        """Update Pyomo model with sensor positions and solve"""

        solver = SolverFactory('glpk')
        solver.solve(self.model)

        # Get the optimized sensor positions from the model
        optimised_sensor_positions = [
            [self.model.sensor_positions[i, j].value for j in range(3)
             ] for i in range(len(self.comparison_pts))]
        
        return optimised_sensor_positions

class RBFBlockTrue(Block):
    def __init__(self, center_points, weights, gamma):
        self.center_points = center_points
        self.weights = weights
        self.gamma = gamma
        self.create_rbf_model()

    def create_rbf_model(self):
        """
        Creates an approximation of the external, 'true' RBF regression method
        inside pyomo model. Uses the rbf parameters passed in from the
        originally fitted RBF model. Necessary to be able to move sensors to
        any point in the mesh and sample field value there as we cannot bring
        in the RBF model from outside.
        """

        # Prediction variables
        self.predict = Var(range(len(self.center_points)), within=NonNegativeReals)

        # Minimise squared error between predicted and target values (weights)
        self.obj = Objective(expr=sum(
            (self.predict[i] - self.weights[i])**2 for i in range(
                len(self.center_points))))

        # Calculate prediction using the RBF kernel between the center points
        def rbf_kernel_expr(model, i, j):
            x1, y1, z1 = model.center_points[i]
            x2, y2, z2 = model.center_points[j]
            return math.exp(-model.gamma * ((x1 - x2) ** 2 + (y1 - y2) ** 2 + (z1 - z2) ** 2))  # RBF kernel in 3D
        
        self.rbf_kernel = Param(range(len(self.center_points)),
                                range(len(self.center_points)), initialize={
            (i, j): rbf_kernel_expr(self, i, j) for i in range(
                len(self.center_points)) for j in range(len(self.center_points))
        })
        
        # Prediction: sum of weights * kernel between the center points
        def prediction_rule(predict, weights, rbf_kernel, i):
            """Get RBF regression prediction"""
            return predict[i] == sum(
                weights[j] * rbf_kernel[i, j] for j in range(len(self.center_points)))

        self.prediction_constraints = Constraint(range(len(
            self.center_points)), rule=prediction_rule)

    def solve(self):
        solver = SolverFactory('ipopt')
        solver.solve(self)
        return [self.predict[i].value for i in range(len(self.center_points))]
    
    def predict(self, point):
        """
        Predict value for a given point using the RBF kernel, weights, and gamma.
        """
        prediction = 0
        for i in range(len(self.center_points)):
            center_point = self.center_points[i]
            # Get RBF kernel between given point and center_point
            kernel_value = math.exp(-self.model.gamma * sum(
                (point[j] - center_point[j])**2 for j in range(3)))
            prediction += self.weights[i] * kernel_value
        return prediction    

# Define the RBF Regression Block for sample points, will optimise gamma and weights
class RBFBlockSample(Block):
    def __init__(self, center_points):
        self.n_points = len(self.center_points)
        self.center_points = center_points
        self.create_rbf_model()

    def create_rbf_model(self):
        """Function to create an rbf regression model from the given
        'center_points' (these are the sample points in the higher model),
        does not expect gamma value or weights to be given.
        """
        
        # These variables will be optimised as part of the main pyomo model
        self.gamma = Var(within=NonNegativeReals, initialize=1.0)
        self.weights = Var(range(self.n_points), within=NonNegativeReals, initialize=1.0)
        self.center_points = Var(range(self.n_points), range(3), within=NonNegativeReals) 

        # x is the predicted values at center points (the sample points)
        self.predict = Var(range(self.n_points), within=NonNegativeReals)

        # Minimise squared error between predicted values and target values (
        # weights, NOT the true values, that comparison is done in main model)
        self.obj = Objective(expr=sum(
            (self.predict[i] - self.weights[i])**2 for i in range(self.n_points)))

        # Get rbf kernel
        self.rbf_kernel = Param(range(self.n_points),
                                range(self.n_points),
                                initialize={
            (i, j): rbf_kernel_expr(self, i, j) for i in range(
                self.n_points) for j in range(self.n_points)
        })

        # Get predictions
        self.prediction_constraints = Constraint(range(self.n_points),
                                                 rule=prediction_rule)        

        def rbf_kernel_expr(model, i, j):
            """ Get RBF kernel
            Args:

            Returns:
            """

            x1, y1, z1 = model.center_points[i, 0],
            model.center_points[i, 1],
            model.center_points[i, 2]
            
            x2, y2, z2 = model.center_points[j, 0],
            model.center_points[j, 1],
            model.center_points[j, 2]
            
            return exp(-model.gamma * (
                (x1 - x2) ** 2 + (y1 - y2) ** 2 + (z1 - z2) ** 2))

        def prediction_rule(model, i):
            """ Calculate predictions for field values at point i
            Args:

            Returns:
            """
            return model.predict[i] == sum(
                model.weights[j] * model.rbf_kernel[i, j] for j in range(
                    self.n_points))

    def solve(self):
        solver = SolverFactory('ipopt')
        solver.solve(self)

        solved_weights = [self.weights[i].value for i in range(self.n_points)]
        predictions = [self.predict[i].value for i in range(self.n_points)]
        positions = [[self.center_points[i, j].value for j in range(3)] for i in range(self.n_points)]

        return self.gamma.value, solved_weights, predictions, positions


# if __name__ == "__main__":
#     optimiser = SensorArrayOptimiser(MeshReader.mesh, num_sensors=10)
#     optimiser.define_sensor_objective(field_function, [Sensor.PointSensor, Sensor.PointSensor])
#     optimised_positions = optimiser.evaluate()
