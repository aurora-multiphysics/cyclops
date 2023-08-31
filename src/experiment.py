from optimisers import Problem, Optimiser
from sensor_group import SensorSuite
from fields import Field
import numpy as np




class Experiment():
    """
    Manages the optimisers, true field and sensor suite.
    3 main functions.
    1. Initialisation to define experiment parameters.
    2. Planning to prepare for optimisation.
    3. Design to optimise the experiment.
    """
    def __init__(self, true_field :Field, comparison_pos :np.ndarray[float], optimiser :Optimiser) -> None:
        """
        Args:
            true_field (Field): the field we got from the simulation - that we expect to measure in an experiment.
            comparison_pos (np.ndarray[float]): the positions we will use to compare the true field to the predicted sensor-based field.
            optimiser (Optimiser): the optimiser we will use to design the experiment.
        """
        self.__true_field = true_field
        self.__num_dim = true_field.get_dim()

        self.__comparison_pos = comparison_pos
        self.__comparison_values = true_field.predict_values(self.__comparison_pos)
        self.__num_pos = len(comparison_pos)

        self.__optimiser = optimiser
        self.__sensor_suite = None
        self.__problem = None
        self.__keys = None
        self.__active_sensors = None
        self.__repetitions = None
        self.__loss_limit = None
        self.__min_active = None


    def plan_soo(self, sensor_suite :SensorSuite, sensor_bounds :np.ndarray[float], repetitions=10) -> None:
        """
        Prepare for single-objective optimisation.

        Args:
            sensor_suite (SensorSuite): describes which sensors we will use for the experiment.
            sensor_bounds (np.ndarray[float]): bounds within which a sensor can be placed.
            repetitions (int, optional): number of repetitions we will average error over. Defaults to 10.
        """
        self.__sensor_suite = sensor_suite
        num_sensors = sensor_suite.get_num_sensors()
        self.__active_sensors = np.array([True] * num_sensors)
        self.__problem = self.__build_problem(sensor_bounds, num_sensors, 1, self.calc_SOO_loss)
        self.__repetitions = repetitions


    def plan_moo(self, sensor_suite :SensorSuite, sensor_bounds :np.ndarray[float], repetitions=1000, loss_limit=80, min_active=3) -> None:
        """
        Prepare for multi-objective optimisation.

        Args:
            sensor_suite (SensorSuite): describes which sensors we will use for the experiment.
            sensor_bounds (np.ndarray[float]): bounds within which a sensor can be placed.
            depth (int, optional): how many sensors will fail at most. Defaults to 3.
            repetitions (int, optional): number of repetitions we will average error over. Defaults to 10.
            loss_limit (_type_, optional): maximum MSE for a successful experiment. Defaults to 80.
        """
        self.__sensor_suite = sensor_suite
        num_sensors = sensor_suite.get_num_sensors()
        self.__active_sensors = np.array([True] * num_sensors)
        self.__problem = self.__build_problem(sensor_bounds, num_sensors, 2, self.calc_MOO_loss)

        self.__keys = self.__sensor_suite.calc_keys(repetitions)
        self.__repetitions = repetitions
        self.__loss_limit = loss_limit
        self.__min_active = min_active


    def __build_problem(self, sensor_bounds :np.ndarray[float], num_sensors :int, num_obj :int, loss_function :callable) -> Problem:
        """
        Args:
            sensor_bounds (np.ndarray[float]): bounds within which a sensor can be placed.
            num_sensors (int): number of sensors used.
            num_obj (int): number of objectives to optimiser for.
            loss_function (callable): loss function to minimse.

        Returns:
            Problem: problem object to optimise.
        """
        low_border = list(sensor_bounds[0]) * num_sensors
        high_border = list(sensor_bounds[1]) * num_sensors
        return Problem(
            num_dim=num_sensors*self.__num_dim,
            num_obj=num_obj,
            loss_function=loss_function,
            borders=[low_border, high_border]
        )

    
    def design(self) -> any:
        """
        Returns:
            any: results object containing Pareto-optimal layouts and optimisation history.
        """
        return self.__optimiser.optimise(self.__problem)


    def calc_MOO_loss(self, sensor_array :np.ndarray[float]) -> list[float]:
        sensor_pos = sensor_array.reshape(-1, self.__num_dim)
        losses = np.zeros(self.__repetitions)
        for i, key in enumerate(self.__keys):
            num_active = np.count_nonzero(key == True)
            if num_active >= self.__min_active:
                self.__active_sensors = key
                losses[i] = self.get_MSE(sensor_pos)
            else:
                losses[i] = np.argmax(losses)

        expected_loss = np.mean(losses)
        failure_chance = (losses>self.__loss_limit).sum()/self.__repetitions
        return [expected_loss, failure_chance]


    
    def calc_SOO_loss(self, sensor_array :np.ndarray[float]) -> list[float]:
        """
        Calculates loss for SOO.

        Args:
            sensor_array (np.ndarray[float]): unshaped sensor layout from optimiser

        Returns:
            list[float]: _description_
        """
        sensor_pos = sensor_array.reshape(-1, self.__num_dim)
        loss = []
        for i in range(self.__repetitions):
            loss.append(self.get_MSE(sensor_pos))
        return [sum(loss)/len(loss)]


    def get_MSE(self, sensor_pos :np.ndarray[float]) -> float:
        """
        Calculate the MSE from an array of proposed sensor positions.
        1. Update the sensor suite to the values at those positions.
        2. See what the sensor suite predicts the rest of the field would be.
        3. Calcualte MSE.

        Args:
            sensor_pos (np.ndarray[float]): n by d array of n sensor positions of d dimensions.

        Returns:
            float: the MSE.
        """
        predict_pos = self.__sensor_suite.get_predict_pos(sensor_pos, self.__active_sensors)
        sensor_values = self.__true_field.predict_values(predict_pos)

        measured_values = self.__sensor_suite.set_sensor_values(sensor_values, self.__active_sensors)
        self.__sensor_suite.fit_sensor_model(sensor_pos, measured_values)
        predicted_values = self.__sensor_suite.predict_data(self.__comparison_pos)

        differences = np.square(predicted_values - self.__comparison_values)
        return np.sum(differences)/self.__num_pos


    def get_SOO_plotting_arrays(self, sensor_array :np.ndarray[float]) -> tuple:
        """
        Finds the necessary data to plot graphs of the potential sensor setup.

        Args:
            sensor_array (np.ndarray[float]): array of unshaped sensor positions from optimiser.

        Returns:
            tuple: Contains all plotting arrays needed.
        """
        sensor_pos = sensor_array.reshape(-1, self.__num_dim)
        predict_pos = self.__sensor_suite.get_predict_pos(sensor_pos, self.__active_sensors)
        sensor_values = self.__true_field.predict_values(predict_pos)

        measured_values = self.__sensor_suite.set_sensor_values(sensor_values, self.__active_sensors)
        self.__sensor_suite.fit_sensor_model(sensor_pos, measured_values)
        predicted_values = self.__sensor_suite.predict_data(self.__comparison_pos)

        differences = np.square(predicted_values - self.__comparison_values)
        print('Loss shown:', np.sum(differences)/self.__num_pos)
        return sensor_pos, self.__comparison_values, predicted_values, measured_values


    def get_MOO_plotting_arrays(self, sensor_array :np.ndarray[float]) -> tuple:
        num_failures = 0
        num_successes = 0
        losses = np.zeros(self.__repetitions)
        all_predicted_values = []
        all_measured_values = []

        for i, key in enumerate(self.__keys):
            num_active = np.count_nonzero(key == True)
            if num_active >= self.__min_active:
                self.__active_sensors = key
                sensor_pos, comp, predicted_values, measured_values = self.get_SOO_plotting_arrays(sensor_array)
                all_predicted_values.append(predicted_values)
                all_measured_values.append(measured_values)

        num_failures = (losses>self.__loss_limit).sum()
        num_successes = self.__repetitions - num_failures
        return sensor_pos, self.__comparison_values, all_predicted_values, all_measured_values, num_failures, num_successes