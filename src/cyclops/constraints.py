import meshio
import numpy as np
import cyclops.sensors as Sensor
import cyclops.sensor_suite as SensorSuite
from pyomo.environ import *
from cyclops.sim_reader import MeshReader
from random import shuffle
from cyclops.regressors import RegressionModel
from cyclops.fields import Field

# ToDo allow for no. of sensors to vary, should take the form of an
# additional constraint, something like:
# model.max_sensors = Constraint(expr=sum(
# model.num_sensors[i] for i in range(1, 6)) <= n)  # Max of n sensors allowed
# model.min_sensors = Constraint(expr=sum(
# model.num_sensors[i] for i in range(1, 6)) >= n)  # Min of m sensors allowed


# class SensorPlacementOptimisation:
#     """"Class to optimise the placement of sensors on a given mesh"""
#     def __init__(self, mesh, num_sensors, sensor_types, min_distance=0.5):
#         """Initialise a Pyomo model and begin populating it with a given mesh,
#         sensor data and constraints."""
#         self.__mesh = mesh
#         self.__num_sensors = num_sensors
#         self.__min_distance = min_distance
#         self.__sensor_types = sensor_types
        
#         # Initialize the Pyomo model
#         self.__model = ConcreteModel()
        
#         # ToDo check for a more optimised way of doing this, i.e. so only
#         # points on surfaces are even considered 
#         # Define sensor positions (x, y, z coordinates)
#         self.__model.x = Var(range(self.__num_sensors),
#                              domain=NonNegativeReals)
#         self.__model.y = Var(range(self.__num_sensors),
#                              domain=NonNegativeReals)
#         self.__model.z = Var(range(self.__num_sensors),
#                              domain=NonNegativeReals)
        
#         # ToDo edit so that the faces covered are surface ONLY
#         # Define lambda variables for barycentric coordinates (vertex weights)
#         self.__model.lambdas = Var(range(self.__num_sensors), range(
#             self.__mesh.__num_faces), domain=NonNegativeReals)
        
#         # Constraints: Sensor positions must lie on the surface of the mesh
#         self.__model.sensor_constraints = Constraint(range(
#             self.__num_sensors), rule=self._sensor_on_surface)
        
#         # Constraints: Barycentric coordinates must sum to 1 for each sensor
#         self.__model.barycentric_constraints = Constraint(range(
#             self.__num_sensors), rule=self._barycentric_sum_to_one)
        
#         # Constraints: Sensors should be at least min_distance apart
#         self.__model.sensor_dist_constraints = Constraint(range(
#             self.__num_sensors), range(self.__num_sensors),
#             rule=self._sensor_distance_constraint)
        
#     def _sensor_on_surface(self, model, sensor_idx, bounding_faces):
#         """Function to ensure the sensor lies on the surface of the mesh and
#         cannot be free-floating inside the mesh. This is done using
#         barycentric coordinates)."""
        
#         # use shuffle to randomise sensor-face pairings
#         shuffle(bounding_faces)

#         # ToDO: this won't prevent sensors being placed on the same face, but
#         # it is unlikely
#         chosen_face = bounding_faces[sensor_idx]
#         face_vertices = self.__mesh.get_face_vertices(chosen_face)
        
#         # Get the 3D coordinates of the face's vertices
#         nodes = [self.__mesh.get_node(v) for v in face_vertices]
        
#         # Variables for extracting (xyz) positions for centre of the sensor
#         x_pos = model.x[sensor_idx]
#         y_pos = model.y[sensor_idx]
#         z_pos = model.z[sensor_idx]

#         # Generalize barycentric coordinate calculation: sum of (lambda_i *
#         # vertex_coordinates_i) where lambda_i are the barycentric coordinates
#         # for each vertex of the face. Barycentric coordinates should be
#         # applied to x, y, and z separately for each coordinate.

#         # The calculation has the form:
#         # x_pos = λ1 * x1 + λ2 * x2 +...+ λn * xn for an n-sided polygon.
#         # repeated similarly for the y and z coords.

#         # Note the & is Pyomo syntax, it is overloaded to allow us to combine
#         # multiple constraint expressions into one
#         return (
#             sum(model.lambdas[sensor_idx, chosen_face, i] * nodes[i][0] for i
#                 in range(len(face_vertices))) == x_pos
#         ) & (
#             sum(model.lambdas[sensor_idx, chosen_face, i] * nodes[i][1] for i
#                 in range(len(face_vertices))) == y_pos
#         ) & (
#             sum(model.lambdas[sensor_idx, chosen_face, i] * nodes[i][2] for i
#                 in range(len(face_vertices))) == z_pos
#         )
        
#     def _barycentric_sum_to_one(self, model, sensor_idx):
#         """Ensure the sum of the barycentric coords equals 1."""
#         return sum(model.lambdas[sensor_idx, face_idx] for face_idx in range(
#             self.__mesh.__num_faces)) == 1
        
#     def _sensor_distance_constraint(self, model, sensor_idx_i, sensor_idx_j):
#         """Ensure sensors are at least `min_distance` apart."""
#         if sensor_idx_i < sensor_idx_j:  # Enforces for each unique pair
#             x_i, y_i, z_i = model.x[sensor_idx_i], model.y[sensor_idx_i],
#             model.z[sensor_idx_i]
            
#             x_j, y_j, z_j = model.x[sensor_idx_j], model.y[sensor_idx_j],
#             model.z[sensor_idx_j]

#             distance = np.sqrt((x_i - x_j)**2 + (y_i - y_j)**2 + (
#                 z_i - z_j)**2)
            
#             return distance >= self.__min_distance
        
#         return Constraint.Skip
        
#     def define_sensor_objective(self, true_field : Field, sensors):
#         """Define the objective function to minimize the field reconstruction
#         error.
        
#         Args:
#             true_field (cyclops field): the 'ground truth' field being sampled
#             and compared against
#             sensors (list of cyclops sensors): a list of the sensor types that
#             will be used in the simulation, these types should be initialised
#             instances of those sensors.
#         Returns:
#             self.model.obj (pyomo objective): this is the objective function
#             that will be solved by pyomo to obtain an answer to the overall
#             optimisation problem.
#         """
#         def error_function(model, true_field):
#             """
#             Calculate the error on the measured field compared to the true 
#             field values.

#             Args:

#             Returns:

#             """
#             error = 0

#             sensor_pos = []

#             for sensor_idx in range(self.__num_sensors):

#                 x_s = self.__model.x[sensor_idx] # coords of sensor_idx
#                 y_s = self.__model.y[sensor_idx]
#                 z_s = self.__model.z[sensor_idx]

#                 # Get current sensor type 
#                 sensor_type = self.__sensor_types[sensor_idx]
#                 sensor_pos.append([x_s, y_s, z_s])

#             #initialise the sensor suite
#             sensor_suite = SensorSuite(true_field, sensor_type, sensor_pos)
#             # Get positions of readings and the readings themselves
#             sensor_sites, readings = sensor_suite.get_sensor_sites(sensor_pos)
#             site_values = sensor_suite.__true_field.predict_values(sensor_sites)
#             # Fit the measured data to a regression model
#             sensor_suite.fit_sensor_model(site_values)

#             true_field_value = true_field(self.__comparison_values)

#             # Get the predicted field values for the reconstruction at the
#             # pre-determined comparison points
#             predicted_measurements = sensor_suite.predict_data(
#                 self.__comparison_pos)
            
#             # Minimize squared error #Todo: is this the right thing for minimising?
#             error += np.mean(np.square(predicted_measurements - true_field_value))

        
#         #ToDo not 100% sure it should be a "rule" and not an "expr"?
#         self.__model.obj = Objective(rule=error_function, sense=minimize)

#     def update_sensor_positions(self, sensor_positions):
#         """Update sensor positions in the model before attempting solve."""
#         for i, (x, y, z) in enumerate(sensor_positions):
#             self.__model.x[i].value = x
#             self.__model.y[i].value = y
#             self.__model.z[i].value = z

#     def evaluate(self, sensor_positions):
#         """Update model, run optimisation problem, and return the objectives."""
#         self.update_sensor_positions(sensor_positions)
#         self.solve()
        
#         # objectives (MSE, failure probability, etc.)
#         obj_values = [self.__model.obj()]
#         return np.array(obj_values)   

#     def solve(self):
#         """Solve the optimisation problem."""
#         solver = SolverFactory('ipopt')
#         solver.solve(self.__model, tee=True)
        
#         # Extract the optimised sensor positions
#         sensor_positions = [(self.__model.x[i].value, self.__model.y[i].value,
#                              self.__model.z[i].value) for i in range(
#                                  self.__num_sensors)]
#         return sensor_positions


# # Initialize optimisation problem #ToDo this doesn't belong here
# optimiser = SensorPlacementOptimisation(MeshReader.mesh, num_sensors=10)

# # Set the objective function
# optimiser.define_sensor_objective(field_function,[Sensor.PointSensor, Sensor.PointSensor])

# # Solve the optimisation problem
# optimised_positions = optimiser.evaluate()

