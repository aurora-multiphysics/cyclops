from optimisers import Problem, Optimiser
from sensor_group import SensorSuite
from fields import Field
import numpy as np




class Experiment():
    def __init__(self, true_field :Field, comparison_pos :np.ndarray, optimiser :Optimiser) -> None:
        self.__true_field = true_field
        self.__num_dim = true_field.get_dim()

        self.__comparison_pos = comparison_pos
        self.__comparison_values = true_field.predict_values(self.__comparison_pos)
        self.__num_pos = len(comparison_pos)

        self.__optimiser = optimiser
        self.__sensor_suite = None
        self.__problem = None
    

    def plan_soo(self, sensor_suite :SensorSuite, sensor_bounds :np.ndarray) -> None:
        self.__sensor_suite = sensor_suite
        num_sensors = sensor_suite.get_num_sensors()
        self.__problem = self.__build_problem(sensor_bounds, num_sensors, 1)


    def plan_moo(self, sensor_suite :SensorSuite, sensor_bounds :np.ndarray) -> None:
        self.__sensor_suite = sensor_suite
        num_sensors = sensor_suite.get_num_sensors()
        self.__problem = self.__build_problem(sensor_bounds, num_sensors, 2)


    def __build_problem(self, sensor_bounds :np.ndarray, num_sensors :int, num_obj :int) -> Problem:
        low_border = list(sensor_bounds[0]) * num_sensors
        high_border = list(sensor_bounds[1]) * num_sensors
        return Problem(
            num_dim=num_sensors*self.__num_dim,
            num_obj=num_obj,
            loss_function=self.get_MSE,
            borders=[low_border, high_border]
        )

    
    def design(self):
        return self.__optimiser.optimise(self.__problem)


    def get_MSE(self, sensor_array :np.ndarray[float]) -> list[float]:
        # Figure out where to find the temperatures
        sensor_pos = sensor_array.reshape(-1, self.__num_dim)
        predict_pos = self.__sensor_suite.get_predict_pos(sensor_pos)

        # Get the sensor values at those positions
        sensor_values = self.__true_field.predict_values(predict_pos)
        self.__sensor_suite.set_sensors(sensor_pos, sensor_values)
        predicted_values = self.__sensor_suite.predict_data(self.__comparison_pos)

        # Calcualte the error
        differences = np.square(predicted_values - self.__comparison_values)
        return [np.sum(differences)/self.__num_pos]


    def get_plotting_arrays(self, sensor_array :np.ndarray[float]) -> tuple:
        # Figure out where to find the temperatures
        sensor_pos = sensor_array.reshape(-1, self.__num_dim)
        predict_pos = self.__sensor_suite.get_predict_pos(sensor_pos)

        # Get the sensor values at those positions
        sensor_values = self.__true_field.predict_values(predict_pos)
        real_values = self.__sensor_suite.set_sensors(sensor_pos, sensor_values)
        predicted_values = self.__sensor_suite.predict_data(self.__comparison_pos)

        return sensor_pos, self.__comparison_values, predicted_values, real_values