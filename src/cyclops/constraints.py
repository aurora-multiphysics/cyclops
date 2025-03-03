import meshio
import numpy as np
import cyclops.sensors as Snsr
from pyomo.environ import *
from cyclops.sim_reader import MeshReader
from random import shuffle
from cyclops.regressors import RegressionModel

# ToDo allow for no. of sensors to vary, should take the form of an
# additional constraint, something like:
# model.max_sensors = Constraint(expr=sum(
# model.num_sensors[i] for i in range(1, 6)) <= n)  # Max of n sensors allowed
# model.min_sensors = Constraint(expr=sum(
# model.num_sensors[i] for i in range(1, 6)) >= n)  # Min of m sensors allowed


class SensorPlacementOptimization:
    """"Class to optimise the placement of sensors on a given mesh"""
    def __init__(self, mesh, num_sensors, sensor_types, min_distance=0.5):
        """Initialise a Pyomo model and begin populating it with a given mesh,
        sensor data and constraints."""
        self.__mesh = mesh
        self.__num_sensors = num_sensors
        self.__min_distance = min_distance
        self.__sensor_types = sensor_types
        
        # Initialize the Pyomo model
        self.__model = ConcreteModel()
        
        # ToDo check for a more optimised way of doing this, i.e. so only
        # points on surfaces are even considered 
        # Define sensor positions (x, y, z coordinates)
        self.__model.x = Var(range(self.__num_sensors),
                             domain=NonNegativeReals)
        self.__model.y = Var(range(self.__num_sensors),
                             domain=NonNegativeReals)
        self.__model.z = Var(range(self.__num_sensors),
                             domain=NonNegativeReals)
        
        # ToDo edit so that the faces covered are surface ONLY
        # Define lambda variables for barycentric coordinates (vertex weights)
        self.__model.lambdas = Var(range(self.__num_sensors), range(
            self.__mesh.__num_faces), domain=NonNegativeReals)
        
        # Constraints: Sensor positions must lie on the surface of the mesh
        self.__model.sensor_constraints = Constraint(range(
            self.__num_sensors), rule=self._sensor_on_surface)
        
        # Constraints: Barycentric coordinates must sum to 1 for each sensor
        self.__model.barycentric_constraints = Constraint(range(
            self.__num_sensors), rule=self._barycentric_sum_to_one)
        
        # Constraints: Sensors should be at least min_distance apart
        self.__model.sensor_dist_constraints = Constraint(range(
            self.__num_sensors), range(self.__num_sensors),
            rule=self._sensor_distance_constraint)
        
    def sensor_on_surface(self, model, sensor_idx, bounding_faces):
        """Function to ensure the sensor lies on the surface of the mesh and
        cannot be free-floating inside the mesh. This is done using
        barycentric coordinates)."""
        
        # use shuffle to randomise sensor-face pairings
        shuffle(bounding_faces)

        # ToDO: this won't prevent sensors being placed on the same face, but
        # it is unlikely
        chosen_face = bounding_faces[sensor_idx]
        face_vertices = self.__mesh.get_face_vertices(chosen_face)
        
        # Get the 3D coordinates of the face's vertices
        nodes = [self.__mesh.get_node(v) for v in face_vertices]
        
        # Variables for extracting (xyz) positions of the sensor
        x_pos = model.x[sensor_idx]
        y_pos = model.y[sensor_idx]
        z_pos = model.z[sensor_idx]

        # Generalize barycentric coordinate calculation: sum of (lambda_i *
        # vertex_coordinates_i) where lambda_i are the barycentric coordinates
        # for each vertex of the face. Barycentric coordinates should be
        # applied to x, y, and z separately for each coordinate.

        # The calculation has the form:
        # x_pos = λ1 * x1 + λ2 * x2 +...+ λn * xn for an n-sided polygon.
        # repeated similarly for the y and z coords.

        # Note the & is Pyomo syntax, it is overloaded to allow us to combine
        # multiple constraint expressions into one
        return (
            sum(model.lambdas[sensor_idx, chosen_face, i] * nodes[i][0] for i
                in range(len(face_vertices))) == x_pos
        ) & (
            sum(model.lambdas[sensor_idx, chosen_face, i] * nodes[i][1] for i
                in range(len(face_vertices))) == y_pos
        ) & (
            sum(model.lambdas[sensor_idx, chosen_face, i] * nodes[i][2] for i
                in range(len(face_vertices))) == z_pos
        )
        
    def barycentric_sum_to_one(self, model, sensor_idx):
        """Ensure the sum of the barycentric coords equals 1."""
        return sum(model.lambdas[sensor_idx, face_idx] for face_idx in range(
            self.__mesh.__num_faces)) == 1
        
    def sensor_distance_constraint(self, model, sensor_idx_i, sensor_idx_j):
        """Ensure sensors are at least `min_distance` apart."""
        if sensor_idx_i < sensor_idx_j:  # Enforces for each unique pair
            x_i, y_i, z_i = model.x[sensor_idx_i], model.y[sensor_idx_i],
            model.z[sensor_idx_i]
            
            x_j, y_j, z_j = model.x[sensor_idx_j], model.y[sensor_idx_j],
            model.z[sensor_idx_j]

            distance = np.sqrt((x_i - x_j)**2 + (y_i - y_j)**2 + (
                z_i - z_j)**2)
            
            return distance >= self.__min_distance
        
        return Constraint.Skip
        
    def define_sensor_objective(self, field_function, true_field):
        """Define the objective function to minimize the field reconstruction error."""
        def objective_function(model):
            error = 0

            # Need to make sure sensor types define exactly what the sensors
            # "see" so this can be used as input for the models in
            # regressors.py, and then the fit data can be used to define
            # field_value
            detected_field = []
            for sensor_idx in range(self.__num_sensors):

                x_s = model.x[sensor_idx] # coords of sensor_idx
                y_s = model.y[sensor_idx]
                z_s = model.z[sensor_idx]

                # Get current sensor type =
                sensor_type = self.__sensor_types[sensor_idx]
                # Call function to get field points observed by sensor
                if sensor_type == 'PointSensor':
                    #ToDo unsure how to establish all the variables here
                    sensor_obvs = Snsr.PointSensor(
                        noise_dev=,offset_function=,failure_chance=,
                        value_range=, field_dim=).get_output_values(
                            site_values=, actual_pos=)

                #ToDo currently regression models can't be directly called here
                # not sure how to handle this, but it needs to be done
                field_value = field_function(sensor_obvs)
                true_field_value = true_field(x_s, y_s, z_s)
                
                # Minimize squared error #Todo: is this the right thing for minimising?
                error += (field_value - true_field_value)**2
            return error
        #ToDo not 100% sure it should be a "rule" and not an "expr"?
        self.__model.obj = Objective(rule=objective_function, sense=minimize)
        
    def solve(self):
        """Solve the optimization problem."""
        solver = SolverFactory('ipopt')
        solver.solve(self.__model, tee=True)
        
        # Extract the optimized sensor positions
        sensor_positions = [(self.__model.x[i].value, self.__model.y[i].value,
                             self.__model.z[i].value) for i in range(
                                 self.__num_sensors)]
        return sensor_positions


# Initialize optimization problem #ToDo this doesn't belong here
optimizer = SensorPlacementOptimization(MeshReader.mesh, num_sensors=10)

# Define a field function (example: a simple function to minimize)
def field_function(sensor_observations):
    #ToDo something to access regressors.py and calculate what the field
    # looks like to the sensors
    return sensor_observations

# Set the objective function
optimizer.define_sensor_objective(field_function)

# Solve the optimization problem
optimized_positions = optimizer.solve()

