from regressors import RBFModel, LModel, GPModel, CSModel, CTModel, PModel
from sensors import Sensor, PointSensor, RoundSensor, Thermocouple
from optimisers import NSGA2Optimiser, PSOOptimiser, GAOptimiser
from sensor_group import SensorSuite, SymmetryManager
from fields import ScalarField, VectorField
from file_reader import PickleManager
from experiment import Experiment
from graphs import GraphManager
import numpy as np




if __name__ == '__main__':
    # Load any objects necessary
    pickle_manager = PickleManager()
    graph_manager = GraphManager()
    true_temp_field = pickle_manager.read_file('simulation', 'field_temp_line.obj')
    grid = pickle_manager.read_file('simulation', 'line_temp.obj')

    bounds = true_temp_field.get_bounds()
    sensor_bounds = bounds + np.array([[1], [-1]])*0.002

    # Setup the sensor suite
    def f(x): return 0
    sensor = Sensor(0, f)
    sensors = np.array([sensor]*5)
    sensor_suite = SensorSuite(
        ScalarField(GPModel, bounds, true_temp_field.get_dim()), 
        sensors,
        symmetry=[]
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
    proposed_layout, true_temps, model_temps, sensor_values = experiment.get_plotting_arrays(res.X)


    # Display the results
    graph_manager.draw(graph_manager.build_optimisation(
        res.history
    ))
    graph_manager.draw(graph_manager.build_1D_compare(
        grid,
        proposed_layout,
        sensor_values,
        true_temps,
        model_temps,
    ))