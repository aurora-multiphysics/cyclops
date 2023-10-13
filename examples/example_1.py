"""
Example script for cyclops.

Demonstration of cyclops used to optimise the positions of a suite of
thermocouples (temperature point sensors) for a thermomechanical experiment
of a divertor monoblock: a cooling component used in fusion devices.

1. Take in a simulation output file and information about the
thermocouple sensor.

2. Emulate the sensor readings by post-processing from the simulation data
according to the sensor properties. Allow some sensors to fail according to
their failure chance.

3. Reconstruct the full temperature field by regression algorithm trained on
just the emulated sensor readings, excluding failed sensors.

4. Optimise the placement of the thermocouples to minimise the difference
between the “true” simulated field and the field reconstructed by the emulated
sensor data.

(c) Copyright UKAEA 2023.
"""
import numpy as np

from cyclops.experiment import Experiment
from cyclops.fields import ScalarField
from cyclops.object_reader import PickleManager
from cyclops.optimisers import NSGA2Optimiser
from cyclops.plotting import PlotManager
from cyclops.regressors import RBFModel
from cyclops.sensor_suite import SensorSuite
from cyclops.sensors import Thermocouple

# Load any objects necessary
pickle_manager = PickleManager()
graph_manager = PlotManager()
true_temp_field = pickle_manager.read_file(
    "tutorials/results/temp_line_field.pickle"
)
grid = pickle_manager.read_file("tutorials/results/temp_line_points.pickle")

field_bounds = true_temp_field.get_bounds()
sensor_bounds = field_bounds + np.array([[1], [-1]]) * 0.002

# Setup the sensor suite
temps = pickle_manager.read_file("sensors/k-type-T.pickle")
voltages = pickle_manager.read_file("sensors/k-type-V.pickle")
sensor = Thermocouple(temps, voltages, 1)
sensors = np.array([sensor] * 5)

sensor_suite = SensorSuite(ScalarField(RBFModel, field_bounds), sensors)

# Setup the experiment
optimiser = NSGA2Optimiser("00:00:10")
experiment = Experiment(true_temp_field, grid, optimiser)
experiment.plan_moo(
    sensor_suite,
    sensor_bounds,
    repetitions=1000,
    loss_limit=1000,
    num_cores=8,
)
res = experiment.design()
pickle_manager.save_file("results/temp_1D_res.pickle", res)

graph_manager.build_pareto(res.F)
graph_manager.draw()

graph_manager.build_pareto(res.F)
graph_manager.save_png("results/pareto.png")

display_str = input("Enter setup to display [Q to quit]: ")
while display_str.isnumeric():
    (
        proposed_layout,
        true_temps,
        model_temps,
        sensor_values,
    ) = experiment.get_SOO_plotting_arrays(res.X[int(display_str)])
    print("\nLoss:", experiment.calc_moo_loss(res.X[int(display_str)]))
    graph_manager.build_1D_compare(
        grid, proposed_layout, sensor_values, true_temps, model_temps
    )
    graph_manager.draw()
    display_str = input("Enter setup to display [Q to quit]: ")
