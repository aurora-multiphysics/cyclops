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
    true_temp_field = pickle_manager.read_file('simulation', 'temp_line_field.obj')
    grid = pickle_manager.read_file('simulation', 'temp_line_points.obj')

    field_bounds = true_temp_field.get_bounds()
    sensor_bounds = field_bounds+np.array([[1], [-1]])*0.002


    # Setup the symmetry
    symmetry_manager = SymmetryManager()
    symmetry_manager.set_1D_x(0.01)

    # Setup the sensor suite
    # temps = pickle_manager.read_file('sensors', 'k-type-T.obj')
    # voltages = pickle_manager.read_file('sensors', 'k-type-V.obj')
    # sensor = Thermocouple(temps, voltages, 1)

    def f(x): return np.zeros(x.shape)
    sensor = MultiSensor(0, f, 0.1, np.array([[-5000], [5000]]), np.linspace(sensor_bounds[0, 0], sensor_bounds[1, 0], 10).reshape(-1, 1))
    sensors = np.array([sensor])

    sensor_suite = SensorSuite(
        ScalarField(CSModel, field_bounds), 
        sensors
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
    graph_manager.build_1D_compare(
        grid,
        proposed_layout,
        sensor_values,
        true_temps,
        model_temps
    )
    graph_manager.draw()