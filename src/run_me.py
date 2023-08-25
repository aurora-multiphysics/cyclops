from regressors import RBFModel, LModel, GPModel, CSModel, CTModel, PModel
from optimisers import NSGA2Optimiser, PSOOptimiser, GAOptimiser
from fields import ScalarField, VectorField
from experiment import Experiment
from file_reader import PickleManager
from sensor_group import SensorSuite
from graphs import GraphManager
from sensors import Sensor
import numpy as np




if __name__ == '__main__':
    pickle_manager = PickleManager()
    graph_manager = GraphManager()
    true_temp_field = pickle_manager.read_file('simulation', 'field_temp_line.obj')
    #grid = pickle_manager.read_file('simulation', 'grid.obj')

    bounds = true_temp_field.get_bounds()
    #sensor_bounds = bounds+np.ones(bounds.shape)*0.002
    sensor_bounds = bounds+np.array([[1], [-1]])*0.001
    grid = np.linspace(bounds[0], bounds[1]).reshape(-1, 1)

    def f(x): return 0
    sensor = Sensor(0, f)
    sensors = np.array([sensor]*3)
    sensor_suite = SensorSuite(
        ScalarField(PModel, bounds, true_temp_field.get_dim()), 
        sensors
    )

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

    proposed_layout, true_temps, model_temps, sensor_values = experiment.get_plotting_arrays(res.X)


    graph_manager.draw(graph_manager.build_optimisation(
        res.history
    ))
    # graph_manager.draw(graph_manager.build_2D_compare(
    #     grid,
    #     proposed_layout,
    #     true_temps,
    #     model_temps
    # ))
    graph_manager.draw(graph_manager.build_1D_compare(
        grid,
        proposed_layout,
        sensor_values,
        true_temps,
        model_temps,
    ))
    print(np.abs(true_temps - model_temps))
    # graph_manager.draw(graph_manager.build_3D_compare(
    #     grid,
    #     true_temps,
    #     model_temps
    # ))