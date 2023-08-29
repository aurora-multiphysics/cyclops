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
        self.__keys = None
        self.__chances = None
        self.__active_sensors = None
        self.__repetitions = None
        self.__loss_limit = None


    def plan_soo(self, sensor_suite :SensorSuite, sensor_bounds :np.ndarray, repetitions=10) -> None:
        self.__sensor_suite = sensor_suite
        num_sensors = sensor_suite.get_num_sensors()
        self.__problem = self.__build_problem(sensor_bounds, num_sensors, 1, self.calc_SOO_loss)
        self.__active_sensors = np.array([True]*num_sensors)
        self.__repetitions = repetitions


    def plan_moo(self, sensor_suite :SensorSuite, sensor_bounds :np.ndarray, depth=3, repetitions=10, loss_limit=1e2) -> None:
        self.__sensor_suite = sensor_suite
        num_sensors = sensor_suite.get_num_sensors()
        self.__problem = self.__build_problem(sensor_bounds, num_sensors, 2, self.calc_MOO_loss)
        self.__keys = self.__sensor_suite.calc_keys(depth)
        self.__chances = self.__sensor_suite.calc_chances(self.__keys)
        self.__repetitions = repetitions
        self.__loss_limit = loss_limit


    def __build_problem(self, sensor_bounds :np.ndarray, num_sensors :int, num_obj :int, loss_function) -> Problem:
        low_border = list(sensor_bounds[0]) * num_sensors
        high_border = list(sensor_bounds[1]) * num_sensors
        return Problem(
            num_dim=num_sensors*self.__num_dim,
            num_obj=num_obj,
            loss_function=loss_function,
            borders=[low_border, high_border]
        )

    
    def design(self):
        return self.__optimiser.optimise(self.__problem)


    def calc_MOO_loss(self, sensor_array :np.ndarray[float]) -> list[float]:
        # Setup the sensor suite to reflect the input array
        sensor_pos = sensor_array.reshape(-1, self.__num_dim)

        losses = np.zeros(len(self.__keys))
        for i, key in enumerate(self.__keys):
            loss = []
            self.__active_sensors = key
            for i in range(self.__repetitions):
                loss.append(self.get_MSE(sensor_pos))
            losses[i] = sum(loss)/len(loss)
        
        success_chance = 0
        for i, sensor_chance in enumerate(self.__chances):
            if losses[i] < self.__loss_limit:
                success_chance += sensor_chance

        return [np.mean(losses * self.__chances), 1-success_chance]

    
    def calc_SOO_loss(self, sensor_array :np.ndarray[float], repetitions=10) -> list[float]:
        # Setup the sensor suite to reflect the input array
        sensor_pos = sensor_array.reshape(-1, self.__num_dim)
        loss = []
        for i in range(repetitions):
            loss.append(self.get_MSE(sensor_pos))
        return [sum(loss)/len(loss)]


    def get_MSE(self, sensor_pos :np.ndarray[float]) -> list[float]:
        # Setup the sensor suite to reflect the input array
        predict_pos = self.__sensor_suite.get_predict_pos(sensor_pos, self.__active_sensors)
        sensor_values = self.__true_field.predict_values(predict_pos)

        measured_values = self.__sensor_suite.set_sensor_values(sensor_values, self.__active_sensors)
        self.__sensor_suite.fit_sensor_model(sensor_pos, measured_values)
        predicted_values = self.__sensor_suite.predict_data(self.__comparison_pos)

        # Calculate the error
        differences = np.square(predicted_values - self.__comparison_values)
        return np.sum(differences)/self.__num_pos


    def set_all_sensors_active(self):
        for i, e in enumerate(self.__active_sensors):
            if e == False:
                self.__active_sensors[i] = True


    def get_plotting_arrays(self, sensor_array :np.ndarray[float]) -> tuple:
        # Setup the sensor suite to reflect the input array
        sensor_pos = sensor_array.reshape(-1, self.__num_dim)
        predict_pos = self.__sensor_suite.get_predict_pos(sensor_pos, self.__active_sensors)
        sensor_values = self.__true_field.predict_values(predict_pos)

        measured_values = self.__sensor_suite.set_sensor_values(sensor_values, self.__active_sensors)
        self.__sensor_suite.fit_sensor_model(sensor_pos, measured_values)
        predicted_values = self.__sensor_suite.predict_data(self.__comparison_pos)

        differences = np.square(predicted_values - self.__comparison_values)
        print('Loss shown:', np.sum(differences)/self.__num_pos)
        return sensor_pos, self.__comparison_values, predicted_values, measured_values