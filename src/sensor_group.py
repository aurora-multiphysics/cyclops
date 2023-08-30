from itertools import combinations
from sensors import Sensor
from fields import Field
import numpy as np



class SymmetryManager():
    """
    Allows assumptions about the symmetry of the temperature field to be built into the predicted temperature field.
    """
    def __init__(self) -> None:
        """
        Initialises the symmetry parameters
        """
        self.__x_point = 0
        self.__x_line = 0
        self.__y_line = 0
        self.__grad = 0

    
    def set_1D_x(self, value :float) -> None:
        """
        Args:
            value (float): the reflection point in 1D.
        """
        self.__x_point = value

    
    def set_2D_x(self, value :float) -> None:
        """
        Args:
            value (float): the reflection x-axis point in 2D for reflection parallel to the x axis.
        """
        self.__x_line = value

    
    def set_2D_y(self, value :float) -> None:
        """
        Args:
            value (float): the reflection y-axis point in 2D for reflection parallel to the y axis.
        """
        self.__y_line = value


    def set_2D_grad(self, value :float) -> None:
        """
        Args:
            value (float): the gradient of the line through the origin for reflection in that line.
        """
        self.__grad = value


    def reflect_1D(self, x_pos :np.ndarray[float]) -> np.ndarray[float]:
        """
        Reflects an array of positions about the reflection axis. 
        Returns both reflected and original positions.

        Args:
            x_pos (np.ndarray[float]): array of n 1D points.

        Returns:
            np.ndarray[float]: array of 2n 1D points.
        """
        axis = np.ones(x_pos.shape)*self.__x_point
        reflected_arr = 2*axis - x_pos
        return np.concatenate((x_pos, reflected_arr), axis=0)


    def reflect_2D_horiz(self, pos :np.ndarray[float]) -> np.ndarray[float]:
        """
        Reflects an array of positions parallel to the x axis. 
        Returns both reflected and original positions.

        Args:
            pos (np.ndarray[float]): array of n 2D original points.

        Returns:
            np.ndarray[float]: array of 2n 2D points.
        """
        axis = np.ones(len(pos))*self.__x_line
        reflected_arr = np.copy(pos)
        reflected_arr[:, 0] = 2*axis - pos[:, 0]
        return np.concatenate((pos, reflected_arr), axis=0)


    def reflect_2D_vert(self, pos :np.ndarray[float]) -> np.ndarray[float]:
        """
        Reflects an array of positions parallel to the y axis. 
        Returns both reflected and original positions.

        Args:
            pos (np.ndarray[float]): array of n 2D original points.

        Returns:
            np.ndarray[float]: array of 2n 2D points.
        """
        axis = np.ones(len(pos))*self.__y_line
        reflected_arr = np.copy(pos)
        reflected_arr[:, 1] = 2*axis - pos[:, 1]
        return np.concatenate((pos, reflected_arr), axis=0)


    def reflect_2D_line(self, pos :np.ndarray[float]) -> np.ndarray[float]:
        """
        Reflects an array of positions in the line that passes through the origin with gradient as specified. 
        Returns both reflected and original positions.

        Args:
            pos (np.ndarray[float]): array of n 2D original points.

        Returns:
            np.ndarray[float]: array of 2n 2D points.
        """
        m = self.__grad
        reflect_matrix = 1/(1+m**2)*np.array([[1 - m**2, 2*m], [2*m, m**2 - 1]])
        reflected_arr = np.apply_along_axis(reflect_matrix.dot, 0, pos.T)
        return np.concatenate((pos, reflected_arr.T), axis=0)




class SensorSuite():
    """
    The sensor suite contains a collection of sensors and a field object.
    It manages the sensors (figures out how many measurements to take per sensor) and sets their values.
    It records the sensor measured values.
    It predicts the values at various points using its field object.
    """
    def __init__(self, field :Field, sensors :list[Sensor], symmetry=[]) -> None:
        """
        Args:
            field (Field): Field object containing the regression model to allow predictions based off sensor values.
            sensors (list[Sensor]): A list of the sensors we will use to predict the field.
            symmetry (list, optional): A list of assumptions we make about the predicted field's symmetry. Defaults to [].
        """
        self.__field = field
        self.__sensors = sensors
        self.__num_sensors = len(self.__sensors)
        self.__symmetry = symmetry
        self.__bounds = field.get_bounds()
        self.__num_dim = field.get_dim()
        

    def get_predict_pos(self, sensor_pos :np.ndarray[float], active_sensors :np.ndarray[bool]) -> np.ndarray[float]:
        """
        Finds out which positions we need to provide the field values for so that we can calculate the sensor measurements.

        Args:
            sensor_pos (np.ndarray): positions of the sensors.
            active_sensors (np.ndarray): describes which sensors are working (not broken).

        Returns:
            np.ndarray[float]: array of positions to find the field values at.
        """
        record_pos = []
        for i, sensor in enumerate(self.__sensors):
            if active_sensors[i] == True:
                area = sensor.get_area(self.__num_dim)
                positions = sensor_pos[i]*np.ones(area.shape) + area
                for pos in positions:
                    record_pos.append(pos)
        return np.array(record_pos)


    def fit_sensor_model(self, sensor_pos :np.ndarray[float], measured_values :np.ndarray[float]) -> None:
        """
        Given the positions of the sensors and the values measured by the sensors we produce a model of the field.

        Args:
            sensor_pos (np.ndarray): n by d array of the n sensor positions in d dimensions.
            measured_values (np.ndarray): n by m array of the sensor values (m=1 for scalars).
        """
        for transformation in self.__symmetry:
            sensor_pos = transformation(sensor_pos)
            measured_values = np.concatenate((measured_values, measured_values), axis=0)
        new_pos, new_values = self.filter(sensor_pos, measured_values)
        self.__field.fit_model(new_pos, new_values)


    def get_num_sensors(self) -> int:
        """
        Returns:
            int: number of sensors.
        """
        return self.__num_sensors

    
    def set_sensor_values(self, sensor_values :np.ndarray[float], active_sensors :np.ndarray[bool]) -> np.ndarray[float]:
        """
        From the true values in the field, the measured values of the sensors are calculated.

        Args:
            sensor_values (np.ndarray[float]): n by m array of the true values in the field (m=1 for scalars).
            active_sensors (np.ndarray[bool]): describes which sensors are working (not broken).

        Returns:
            np.ndarray[float]: array of the measured sensor values.
        """
        measured_values = np.zeros(self.__num_sensors).reshape(-1, 1)
        index = 0
        for i, sensor in enumerate(self.__sensors):
            if active_sensors[i] == True:
                relevant_values = sensor_values[index:index+sensor.get_num_values(), 0]
                index += sensor.get_num_values()
                sensor.set_value(relevant_values)
                measured_values[i, 0] = sensor.get_value()
        return measured_values


    def predict_data(self, pos :np.ndarray[float]) -> np.ndarray[float]:
        """
        Args:
            pos (np.ndarray[float]): n by d array of n positions of d dimensions to predict the values of

        Returns:
            np.ndarray[float]: n by m array of n values of m dimensions.
        """
        return self.__field.predict_values(pos)

    
    def filter(self, pos_array :np.ndarray[float], measured_values :np.ndarray[float]) -> tuple[np.ndarray[float]]:
        """
        Filters through the positions provided by the symmetry manager to prevent any values that are out of bounds from affecting the model of the field.
        This means that if you choose the symmetry line badly it may not produce an actually symmetric field.

        Args:
            pos_array (np.ndarray[float]): n by d array of n positions with d dimensions to be filtered.
            measured_values (np.ndarray[float]): n by m array of n values with m dimensions that correspond to the positions.

        Returns:
            tuple[np.ndarray[float]]: tuple of the filtered positions and values.
        """
        out_pos = []
        out_value = []
        condition_1 = pos_array > self.__bounds[0]
        condition_2 = pos_array < self.__bounds[1]
        for i, pos in enumerate(pos_array):
            if condition_1[i].all() == True and condition_2[i].all() == True:
                out_pos.append(pos_array[i])
                out_value.append(measured_values[i])
        return np.array(out_pos), np.array(out_value)


    def calc_keys(self, depth :int) -> np.ndarray[bool]:
        """
        Calculate an array of active sensors describing all the combinations of active sensors specified where less than or equal to 'depth' sensors have failed.

        Args:
            depth (int): the maximum number of sensors we consider to have failed.

        Returns:
            np.ndarray[bool]: an array of all the potential active sensors arrays.
        """
        template = np.array(range(0, self.__num_sensors))
        failed_keys = []
        for i in range(depth+1):
            failed_indices = combinations(template, i)
            for setup in failed_indices:
                key=[True]*self.__num_sensors
                for i in setup:
                    key[i] = False
                failed_keys.append(np.array(key))
        return np.array(failed_keys)


    def calc_chances(self, keys :np.ndarray[bool]) -> np.ndarray[float]:
        """
        Calculates the chance of the layouts where some sensors have failed.

        Args:
            keys (np.ndarray[bool]): array of arrays describing which sensors have failed.

        Returns:
            np.ndarray[float]: array of the chance of each setup occurring.
        """
        chances = []
        for key in keys:
            chance = 1
            for i, not_failed in enumerate(key):
                if not_failed:
                    chance *= (1 - self.__sensors[i].get_failure_chance())
                else:
                    chance *= self.__sensors[i].get_failure_chance()
            chances.append(chance)
        return np.array(chances)