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
                 pos_3D: np.ndarray, true_field: Field, min_distance=0.5):
        """
        Class to set up and handle a pyomo model of sensors positioned on the
        surface of a mesh.

        Args:
        mesh_faces: List of (x, y, z) mesh surface coordinates, to be passed after mesh analysis
        
        Returns:
        """
        self.__mesh_faces = mesh_faces
        #self.__mesh = mesh
        self.__num_sensors = num_sensors
        self.__min_distance = min_distance
        self.__sensors = sensors
        self.__true_field = true_field
        self.__3D_pos = pos_3D
    
        # Initialize the Pyomo model
        self.__model = ConcreteModel()

        # Decision variables
        self.__model.Sensors_pos = RangeSet(self.__num_sensors)
        self.__model.TriangleIndex = Var(self.__model.Sensors_pos, within=NonNegativeIntegers, bounds=(0, len(mesh_faces) - 1))
        self.__model.BaryCoords = Var(self.__model.Sensors_pos, [1, 2, 3], within=UnitInterval)
        self.__model.BaryConstraint = Constraint(self.__model.Sensors_pos, rule=self.barycentric_sum_rule)
        self.__model.SensorPosition = Expression(self.__model.Sensors_pos, rule=self.to_cartesian)
        
        # Constraint: Sensors should be at least min_distance apart
        self.__model.sensor_dist_constraints = Constraint(range(
            self.__num_sensors), range(self.__num_sensors),
            rule=self._sensor_distance_constraint)

        # Objective functions (placeholders for now)
        self.__model.MSE = Objective(expr=0, sense=minimize)
        self.__model.FailureRisk = Objective(expr=0, sense=minimize)
        mse = value(self.__model.MSE)
        risk = value(self.__model.FailureRisk)

    def barycentric_sum_rule(model, i):
        """Ensure that barycentric coordinates sum to 1
        
        Args:
            model (pyomo model): the current pyomo model instance
            i (int): the index corresponding to the coordinates to be checked
            (i.e. coordinates of a sensor)

        Returns:
            the barycentric sum rule in pyomo-suitable form
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

        return v1 * model.BaryCoords[i, 1] + v2 * model.BaryCoords[i, 2] + v3 * model.BaryCoords[i, 3]

    def _sensor_distance_constraint(self, model, sensor_idx_i, sensor_idx_j):
        """Ensure sensors are at least `min_distance` apart.
        
        Args:
            model (pyomo model): the current pyomo model instance
            sensor_idx_i (int): the index of a sensor for distance checking
            sensor_idx_j (int): the index of a sensor for distance checking
        Returns:
            Boolean indicating whether or not distance constraint has been met
        """
        if sensor_idx_i < sensor_idx_j:  # Enforces for each unique pair

            x_i = model.BaryCoords[sensor_idx_i, 1].value
            y_i = model.BaryCoords[sensor_idx_i, 2].value
            z_i = model.BaryCoords[sensor_idx_i, 3].value

            x_j = model.BaryCoords[sensor_idx_j, 1].value
            y_j = model.BaryCoords[sensor_idx_j, 2].value
            z_j = model.BaryCoords[sensor_idx_j, 3].value

            distance = np.sqrt((x_i - x_j)**2 + (y_i - y_j)**2 + (
                z_i - z_j)**2)
            
            return distance >= self.__min_distance
        # if the same sensor is compared to itself we skip the constraint
        return Constraint.Skip

    def define_sensor_objective(self, true_field : Field, sensors):
        """Define the objective function to minimize the field reconstruction
        error.
        
        Args:
            true_field (cyclops field): the 'ground truth' field being sampled
            and compared against
            sensors (list of cyclops sensors): a list of the sensor types that
            will be used in the simulation, these types should be initialised
            instances of those sensors.
        Returns:
            self.model.obj (pyomo objective): this is the objective function
            that will be solved by pyomo to obtain an answer to the overall
            optimisation problem.
        """

        def compute_error(model, true_field: Field):
            """
            Calculate the error on the measured field compared to the true 
            field values.

            Args:

            Returns:

            """
            error = 0

            sensor_pos = []

            for sensor_idx in range(self.__num_sensors):

                x_s = self.__model.x[sensor_idx] # coords of sensor_idx
                y_s = self.__model.y[sensor_idx]
                z_s = self.__model.z[sensor_idx]

                # Get current sensor type 
                current_sensor = self.__sensors[sensor_idx]
                sensor_pos.append([x_s, y_s, z_s])

            #initialise the sensor suite
            sensor_suite = SensorSuite(true_field, self.__sensors, sensor_pos, )
            # Do I actually need a method to move the initialised sensors? Within the sensors class?
            # How else to optimise the position? Otherwise would have to re-initialise sensors each
            # time. 


            # Get positions of readings and the readings themselves
            sensor_sites, readings = sensor_suite.get_sensor_sites(sensor_pos)
            # This is currently not properly associating the different points
            # with the type of sensor, need to alter this so that the error
            # from different sensors is properly applied.
            site_values = sensor_suite.__true_field.predict_values(sensor_sites)
            # Fit the measured data to a regression model
            sensor_suite.fit_sensor_model(site_values)

            true_field_value = true_field(self.__comparison_values)

            # Get the predicted field values for the reconstruction at the
            # pre-determined comparison points
            predicted_measurements = sensor_suite.predict_data(
                self.__comparison_pos)
            
            # Minimize squared error #Todo: is this the right thing for minimising?
            error += np.mean(np.square(predicted_measurements - true_field_value))

            return error

        #ToDo not 100% sure it should be a "rule" and not an "expr"?
        self.__model.obj = Objective(rule=compute_error, sense=minimize)

    def update_sensor_positions(self):
        """Update optimized sensor positions from the Pyomo model after solving."""
        self.optimized_positions = [self.to_cartesian(self.__model, i) for i in
                                    self.__model.Sensors_pos]
        return self.optimized_positions

    def evaluate(self, sensor_positions):
        """Update model, run optimisation problem, and return the objectives."""
        self.update_sensor_positions(sensor_positions)
        self.solve()
        
        # objectives (MSE, failure probability, etc.)
        obj_values = [self.__model.obj()]
        return np.array(obj_values)   

    def solve(self, sensor_indices):
        """Update Pyomo model with sensor positions and solve"""
        # Assign sensors?
        #for i, idx in enumerate(sensor_indices):
        #    self.__model.SensorIndex[i] = idx  

        solver = SolverFactory('glpk')
        solver.solve(self.__model)

        # Get objective values
        mse = self.__model.MSE.expr()
        risk = self.__model.FailureRisk.expr()

        return mse, risk

# if __name__ == "__main__":
#     optimiser = SensorPlacementOptimisation(MeshReader.mesh, num_sensors=10)
#     optimiser.define_sensor_objective(field_function, [Sensor.PointSensor, Sensor.PointSensor])
#     optimised_positions = optimiser.evaluate()
