from regressors import RBFModel, GPModel, NModel, LModel, CTModel, CSModel
from fields import ScalarField, VectorField
from read_results import PickleManager
from manage_sensors import SensorSuite
from sensors import Sensor
import numpy as np

from manage_graphs import GraphManager

from matplotlib import pyplot as plt

class ExperimentDesigner():
    def __init__(self, true_field, sensor_suite, comparison_pos) -> None:
        self.__true_field = true_field
        self.__num_dim = true_field.get_dim()
        self.__sensor_suite = sensor_suite
        self.__comparison_pos = comparison_pos
        self.__comparison_values = true_field.predict_values(self.__comparison_pos)
        self.__num_pos = len(comparison_pos)


    def get_loss(self, sensor_array):
        sensor_pos = sensor_array.reshape(-1, self.__num_dim)
        sensor_values = self.__true_field.predict(sensor_pos)
        self.__sensor_suite.fit(sensor_pos, sensor_values)
        predicted_values = self.__sensor_suite.predict(self.__comparison_pos)
        differences = np.square(predicted_values - self.__comparison_values)
        return np.sum(differences)/self.__num_pos


    def get_temps_for_plotting(self, sensor_array):
        sensor_pos = sensor_array.reshape(-1, self.__num_dim)
        sensor_values = self.__true_field.predict_values(sensor_pos)
        self.__sensor_suite.set_sensors(sensor_pos, sensor_values)
        predicted_values = self.__sensor_suite.predict_data(self.__comparison_pos)
        return predicted_values, self.__comparison_values




if __name__ == '__main__':
    pickle_manager = PickleManager()
    true_temps = pickle_manager.read_file('simulation', 'field_temp.obj')
    grid = pickle_manager.read_file('simulation', 'grid.obj')

    bounds = true_temps.get_bounds()

    def f(x): return 0
    sensor = Sensor(0, f)

    sensor_suite = SensorSuite(
        ScalarField(RBFModel, bounds, 2), 
        [sensor, sensor, sensor, sensor, sensor]
    )
    
    designer = ExperimentDesigner(true_temps, sensor_suite, grid)
    sensor_array = np.array([0.0060719710037205514, 0.01720606611846788, 0.0031532261045091967, 0.005071218456956683, 0.010150584337864806, 0.005487546346333083, 0.0006967046228017783, -0.01013460765738274, 0.0010205573392144066, 0.011203007124517491])
    temps, true_temps = designer.get_temps_for_plotting(sensor_array)

    graph_manager = GraphManager()
    fig = graph_manager.build_double_3D_temp_field(
        grid,
        true_temps,
        temps
    )
    plt.show()
    plt.close()
