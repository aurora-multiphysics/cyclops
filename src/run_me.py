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
    temps = pickle_manager.read_file('sensors', 'k-type-T.obj')
    voltages = pickle_manager.read_file('sensors', 'k-type-V.obj')
    sensor = Thermocouple(temps, voltages)
    sensors = np.array([sensor]*5)
    sensor_suite = SensorSuite(
        ScalarField(CSModel, bounds, true_temp_field.get_dim()), 
        sensors,
        symmetry=[]
    )

    # Setup the experiment
    optimiser = NSGA2Optimiser('00:00:10')
    experiment = Experiment(
        true_temp_field,
        grid,
        optimiser
    )
    experiment.plan_moo(
        sensor_suite,
        sensor_bounds,
        depth=2,
        repetitions=10,
        loss_limit=100
    )
    res = experiment.design()


    # Display and save the results
    for i, setup in enumerate(res.X):
        pickle_manager.save_file('results', 'Layout'+str(i)+'.obj', setup.reshape(-1, true_temp_field.get_dim()))

    graph_manager.build_pareto(
        res.F
    )
    graph_manager.draw()

    display_str = input('Enter setup to display [Q to quit]: ')
    while display_str.isnumeric():
        experiment.set_all_sensors_active()
        proposed_layout, true_temps, model_temps, sensor_values = experiment.get_plotting_arrays(res.X[i])
        graph_manager.build_1D_compare(
            grid,
            proposed_layout,
            sensor_values,
            true_temps,
            model_temps
        )
        graph_manager.draw()
        display_str = input('Enter setup to display [Q to quit]: ')

