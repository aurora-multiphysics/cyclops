from sensors import Sensor, PointSensor, RoundSensor, Thermocouple, MultiSensor
from regressors import RBFModel, LModel, GPModel, CSModel, CTModel, PModel
from optimisers import NSGA2Optimiser, PSOOptimiser, GAOptimiser
from sensor_group import SensorSuite, SymmetryManager
from fields import ScalarField, VectorField
from object_reader import PickleManager
from experiment import Experiment
from graphs import GraphManager
import numpy as np




if __name__ == '__main__':
    # Load any objects necessary
    pickle_manager = PickleManager()
    graph_manager = GraphManager()
    true_temp_field = pickle_manager.read_file('simulation', 'temp_plane_field.obj')
    grid = pickle_manager.read_file('simulation', 'temp_plane_points.obj')

    field_bounds = true_temp_field.get_bounds()
    sensor_bounds = field_bounds+np.array([[1, 1], [-1, -1]])*0.002

    # Setup the symmetry
    symmetry_manager = SymmetryManager()
    symmetry_manager.set_2D_x(np.mean(field_bounds[:, 0]))

    # Setup the sensor suite
    def f(x): return np.zeros(x.shape)
    sensor = RoundSensor(0, f, 0, np.array([[-5000], [5000]]), 0, 2)
    sensors = np.array([sensor]*5)

    # def f(x): return np.zeros(x.shape)
    # sensor = MultiSensor(0, f, 0.1, np.array([[-5000], [5000]]), np.linspace(sensor_bounds[0, 0], sensor_bounds[1, 0], 10).reshape(-1, 2))
    # sensors = np.array([sensor])

    sensor_suite = SensorSuite(
        ScalarField(RBFModel, field_bounds), 
        sensors,
        symmetry=[symmetry_manager.reflect_2D_horiz]
    )

    # Setup the experiment
    optimiser = PSOOptimiser('00:00:10')
    experiment = Experiment(
        true_temp_field,
        grid,
        optimiser
    )
    experiment.plan_soo(
        sensor_suite,
        sensor_bounds
    )
    res = experiment.design()
    proposed_layout, true_temps, model_temps, sensor_values = experiment.get_SOO_plotting_arrays(res.X)

    # Display the results
    graph_manager.build_optimisation(
        res.history
    )
    graph_manager.draw()
    graph_manager.build_2D_compare(
        grid,
        proposed_layout,
        true_temps,
        model_temps
    )
    graph_manager.draw()
    graph_manager.build_3D_compare(
        grid,
        true_temps,
        model_temps
    )
    graph_manager.draw()