import numpy as np
import meshio
import numpy as np
import cyclops.sensors as Sensor
import cyclops.sensor_suite as SensorSuite
from pyomo.environ import *
from cyclops.sim_reader import MeshReader
from random import shuffle
from cyclops.regressors import RegressionModel
from cyclops.fields import Field
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from typing import List

# ToDo allow for no. of sensors to vary, should take the form of an
# additional constraint, something like:
# model.max_sensors = Constraint(expr=sum(
# model.num_sensors[i] for i in range(1, 6)) <= n)  # Max of n sensors allowed
# model.min_sensors = Constraint(expr=sum(
# model.num_sensors[i] for i in range(1, 6)) >= n)  # Min of m sensors allowed

#Need to make sure that the list of faces that is passed are all triangles...any quads or hexes need to be subdivided first!
class SensorPlacementOptimisation:
    def __init__(self, mesh_faces: List, num_sensors, sensors,
                 pos_3D: np.ndarray, true_values, min_distance=0.5,
                 regression_type='RBF', rbf_centers=None, rbf_weights=None,
                 rbf_gamma=None):
        """
        Class to set up and handle a pyomo model of sensors positioned on the
        surface of a mesh.

        Args:
        mesh_faces (List): List of (x, y, z) mesh surface coordinates, to be
                        passed after mesh analysis
        num_sensors (int): Integer (max) number of sensors to be place
        pos_3D (np.ndarray): Numpy array of 3D positions in the field
        true_field_values (np.ndarray): selected comparison points for error
                        caculation, these should come from interpolated field.
        min_distance (float): Minimum required distance between sensors, this
                        MUST be in the same units as the size of sensors and
                        as the Field's length.

        Returns:
        """
        self.__mesh_faces = mesh_faces
        self.__face_vertex_map = {
            i: mesh_faces[i] for i in range(len(mesh_faces))
        }
        self.__num_sensors = num_sensors
        self.__min_distance = min_distance
        self.__sensors = sensors
        self.__true_values = true_values
        self.__3D_pos = pos_3D

        self.rbf_centers = rbf_centers        # List of 3D points
        self.rbf_weights = rbf_weights        # List of floats
        self.rbf_gamma = rbf_gamma            # Float


        # Initialize the Pyomo model
        self.__model = ConcreteModel()
        self._build_model(self)


        # Decision variables
        self.__model.Sensors_pos = RangeSet(0, self.__num_sensors-1)
        self.__model.TriangleIndex = Var(self.__model.Sensors_pos, within=NonNegativeIntegers, bounds=(0, len(mesh_faces) - 1))
        self.__model.BaryCoords = Var(self.__model.Sensors_pos, [1, 2, 3], within=UnitInterval)
        
        # Barycentric coordinate sum constraint applied
        self.__model.BaryConstraint = Constraint(
            self.__model.Sensors_pos,
            rule=lambda model, i: self.barycentric_sum_rule(model,i))

        # Need to store vertex coordinates as a Pyomo Param
        self.__model.FaceVertices = Param(
            self.__model.Sensors_pos, [1, 2, 3], [1, 2, 3],
            initialize=lambda model, i, j, k: self.__face_vertex_map[i][j - 1][k - 1],
            within=Reals
        )



        # Create symbolic expressions for x/y/z
        self.__model.SensorX = Expression(self.__model.Sensors_pos,
            rule=lambda model, i: sum(model.FaceVertices[i, j, 1] * model.BaryCoords[i, j] for j in [1, 2, 3])
        )
        self.__model.SensorY = Expression(self.__model.Sensors_pos,
            rule=lambda model, i: sum(model.FaceVertices[i, j, 2] * model.BaryCoords[i, j] for j in [1, 2, 3])
        )
        self.__model.SensorZ = Expression(self.__model.Sensors_pos,
            rule=lambda model, i: sum(model.FaceVertices[i, j, 3] * model.BaryCoords[i, j] for j in [1, 2, 3])
        )



        self.__model.SensorPosition = Expression(self.__model.Sensors_pos, rule=self.to_cartesian)
        
        # Constraint: Sensors should be at least min_distance apart
        self.__model.sensor_dist_constraints = Constraint(range(
            self.__num_sensors), range(self.__num_sensors),
            rule=self._sensor_distance_constraint)

        # Objective functions
        self.__model.obj = Objective(rule=self.reconstruction_error, sense=minimize)


    def _build_model(self):
        if self.regression_type == 'rbf':
            self._build_rbf_model()
        elif self.regression_type == 'linear':
            self._build_linear_model()
        elif self.regression_type == 'poly2':
            self._build_poly2_model()
        else:
            raise ValueError(f"Unsupported regression type: {self.regression_type}")

    def _build_rbf_model(self):
        """
        Build the pyomo model using RBF regression method inside optimiser
        """
        model = self.__model

        # Sets defining centerpoints and dimensions
        model.D = RangeSet(3)
        model.C = RangeSet(len(self.rbf_centers))

        # Points from the reconstructed field
        model.CenterCoords = Param(
            model.C, model.D, initialize=lambda m, c,
            d: self.rbf_centers[c-1][d-1], mutable=True
        )

        # Weights from the reconstructed field
        model.RBFWeights = Param(
            model.C, initialize=lambda m, c: self.rbf_weights[c-1],
            mutable=True
        )

        # RBF gamma from the reconstructed field
        model.Gamma = Param(initialize=self.rbf_gamma, mutable=True)

        # Calculated the predicted field values at sensor positions
        model.PredictedValue = Expression(model.Sensors_pos,
                                          rule=self.rbf_expr_rule)
        
        # Parameterises the "true values" at the same positions
        model.TrueValue = Param(model.Sensors_pos, initialize={
            i+1: val for i, val in enumerate(self.true_field_values)
        })

        # Sets the objective to be the minimisation of the MSE between the two
        model.MSE = Objective(expr=sum(
            (model.PredictedValue[s] - model.TrueValue[s])**2
            for s in model.Sensors_pos), sense=minimize)

    def rbf_expr_rule(m, s):
        """
        Function to return the predicted value measured at each sensor based
        on RBF. """
        return sum(
            m.RBFWeights[c] * exp(-m.Gamma * sum(
                (m.SensorPosition[s, d] - m.CenterCoords[c, d])**2
                for d in m.D
            )) for c in m.C
        )

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

    def to_cartesian(self, model, i):
        """Convert barycentric coordinates back to cartesian coordinates
        
        Args:
            model (pyomo model): the current pyomo model instance
            i (int): the index corresponding to the coordinates to be
            converted (i.e. coordinates of a sensor)

        Returns:
            3D sensor cartesian sensor coordinates corresponding to input
            barycentric coordinates"""
        # model.TriangleIndex holds the indices of the chosen faces for all
        # sensors, hence select a single entry which corresponds to the face
        # index for a single sensor
        face_index = int(model.TriangleIndex[i].value)
        v1, v2, v3 = self.__mesh_faces[face_index]
        b1 = model.BaryCoords[i, 1]
        b2 = model.BaryCoords[i, 2]
        b3 = model.BaryCoords[i, 3]

        return v1 * b1 + v2 * b2 + v3 * b3

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
        return dist_sq >= self.__min_distance ** 2 

    def reconstruction_error(self, model):
        """Define the objective function to minimize the field reconstruction
        error.
        
        Args:
            true_field_values (cyclops field): the 'ground truth' field being sampled
            and compared against
            sensors (list of cyclops sensors): a list of the sensor types that
            will be used in the simulation, these types should be initialised
            instances of those sensors.
        Returns:
            self.model.obj (pyomo objective): this is the objective function
            that will be solved by pyomo to obtain an answer to the overall
            optimisation problem.
        """
        sensor_pos = [value(self.to_cartesian(model, i)) for i in model.Sensors_pos]
        sensor_pos = np.array(sensor_pos)
        
        return sum((model.true_field_values[k] - model.reconstructed_field[k])**2
                        for k in model.EvalPoints)

    def update_sensor_positions(self):
        """Update optimized sensor positions from the Pyomo model after solving."""
        optimized_positions = [self.to_cartesian(self.__model, i) for i in
                                    self.__model.Sensors_pos]
        return optimized_positions

    def evaluate(self, sensor_positions):
        """Update model, run optimisation problem, and return the objectives."""
        self.update_sensor_positions(sensor_positions)
        self.solve()
        
        # objectives (MSE, failure probability, etc.)
        obj_values = [self.__model.obj()]
        return np.array(obj_values)   

    def solve(self):
        """Update Pyomo model with sensor positions and solve"""
        # Assign sensors?
        #for i, idx in enumerate(sensor_indices):
        #    self.__model.SensorIndex[i] = idx  

        solver = SolverFactory('glpk')
        solver.solve(self.__model)

        return self.update_sensor_positions()

# if __name__ == "__main__":
#     optimiser = SensorPlacementOptimisation(MeshReader.mesh, num_sensors=10)
#     optimiser.define_sensor_objective(field_function, [Sensor.PointSensor, Sensor.PointSensor])
#     optimised_positions = optimiser.evaluate()
