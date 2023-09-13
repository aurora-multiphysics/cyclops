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


        # Setup the sensor suite
    temps = pickle_manager.read_file('sensors', 'k-type-T.obj')
    voltages = pickle_manager.read_file('sensors', 'k-type-V.obj')
    sensor = Thermocouple(temps, voltages, 1)
    sensors = np.array([sensor]*5)

    sensor_suite = SensorSuite(
        ScalarField(RBFModel, field_bounds), 
        sensors
    )


    # Setup the experiment
    optimiser = NSGA2Optimiser('00:50:00')
    experiment = Experiment(
        true_temp_field,
        grid,
        optimiser
    )
    experiment.plan_moo(
        sensor_suite,
        sensor_bounds,
        repetitions=1000,
        loss_limit=1000
    )
    res = experiment.design()
    pickle_manager.save_file('results', 'Temp_1D_res.obj', res)


    graph_manager.build_pareto(res.F)
    graph_manager.draw()

    graph_manager.build_pareto(res.F)
    graph_manager.save_png('results', 'Pareto.png')

    display_str = input('Enter setup to display [Q to quit]: ')
    while display_str.isnumeric():
        proposed_layout, true_temps, model_temps, sensor_values = experiment.get_SOO_plotting_arrays(res.X[int(display_str)])
        print('\nLoss:', experiment.calc_MOO_loss(res.X[int(display_str)]))
        graph_manager.build_1D_compare(
            grid,
            proposed_layout,
            sensor_values,
            true_temps,
            model_temps
        )
        graph_manager.draw()
        display_str = input('Enter setup to display [Q to quit]: ')