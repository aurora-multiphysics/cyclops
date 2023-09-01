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
    true_temp_field = pickle_manager.read_file('simulation', 'disp_plane_field.obj')
    grid = pickle_manager.read_file('simulation', 'disp_plane_points.obj')

    field_bounds = true_temp_field.get_bounds()
    sensor_bounds = field_bounds+np.array([[1, 1], [-1, -1]])*0.002

    # Setup the symmetry
    symmetry_manager = SymmetryManager()
    symmetry_manager.set_2D_x(np.mean(field_bounds[:, 0]))

    # Setup the sensor suite
    def f(x): return 0
    sensor = PointSensor(0, f, 0, np.array([[-5e10, -5e10, -5e10], [5e10, 5e10, 5e10]]), 2)
    sensors = np.array([sensor]*5)
    sensor_suite = SensorSuite(
        VectorField(RBFModel, field_bounds), 
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
    proposed_layout, true_disps, model_disps, sensor_vals = experiment.get_SOO_plotting_arrays(res.X)

    mag_true_disps = np.linalg.norm(true_disps, axis=1).reshape(-1, 1)
    mag_model_disps = np.linalg.norm(model_disps, axis=1).reshape(-1, 1)
    mag_sensor_vals = np.linalg.norm(sensor_vals, axis=1).reshape(-1, 1)

    # Display the results
    graph_manager.build_optimisation(
        res.history
    )
    graph_manager.draw()
    graph_manager.build_2D_compare(
        grid,
        proposed_layout,
        mag_true_disps,
        mag_model_disps
    )
    graph_manager.draw()
    graph_manager.build_3D_compare(
        grid,
        mag_true_disps,
        mag_model_disps
    )
    graph_manager.draw()