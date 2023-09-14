from sensors import Sensor
from fields import Field
import numpy as np

    


class SensorSuite():
    """
    Holds the sensors and allows a field to be predicted from the sensor data.
    """
    def __init__(self, field :Field, sensors :np.ndarray[Sensor], symmetry=[]) -> None:
        """
        Args:
            field (Field): field to use as sensor field.
            sensors (np.ndarray[Sensor]): array of sensors to measure the field.
            symmetry (list, optional): list of symmetry assumptions. Defaults to [].
        """
        self.__field = field
        self.__sensors = sensors
        self.__num_sensors = len(self.__sensors)
        self.__symmetry = symmetry

        self.__active_sensors = np.full(self.__num_sensors, True)
        self.__sensor_pos = np.zeros((self.__num_sensors, self.__field.get_dim()))

    
    def set_active_sensors(self, active_sensors :np.ndarray[bool]):
        """
        Set which sensors are active.

        Args:
            active_sensors (np.ndarray[bool]): array of booleans to show which sensors are off or on.
        """
        self.__active_sensors = active_sensors

    
    def set_sensor_pos(self, sensor_pos :np.ndarray[float]):
        """
        Set the positions of the sensors.

        Args:
            sensor_pos (np.ndarray[float]): n by d array of n positions of d dimensions.
        """
        self.__sensor_pos = sensor_pos


    def get_sensor_sites(self) -> np.ndarray[float]:
        """
        Returns the positions from which the sensors sample from.

        Returns:
            np.ndarray[float]: n by d array of n positions of d dimensions to sample from.
        """
        absolute_sites = []
        for i, sensor in enumerate(self.__sensors):
            sites = sensor.get_input_sites(self.__sensor_pos[i])
            for site in sites:
                absolute_sites.append(site)
        return np.array(absolute_sites)
    

    def __measure_sensor_values(self, site_values :np.ndarray[float]) -> tuple[np.ndarray]:
        """
        Calculates the values measured by the sensors, and the positions those values are thought to be at.

        Args:
            site_values (np.ndarray[float]): s by t by m array of s*t values of dimension m where s is number of sensors.

        Returns:
            tuple[np.ndarray]: contains the n by m array of field values and the n by d array of the positions of those values.
        """
        field_values = []
        field_pos = []
        value_index = 0
        for i, sensor in enumerate(self.__sensors):
            num_input_sites = sensor.get_num_input_sites()
            if self.__active_sensors[i] == True:
                known_values, known_pos = sensor.get_output_values(site_values[value_index:value_index+num_input_sites], self.__sensor_pos[i])
                for i, value in enumerate(known_values):
                    field_values.append(value)
                    field_pos.append(known_pos[i])
            value_index += num_input_sites
        return np.array(field_values), np.array(field_pos)


    def fit_sensor_model(self, site_values :np.ndarray[float]):
        """
        Fit the model based off the sensor data.

        Args:
            site_values (np.ndarray[float]): n by m array of the n values of dimension m at the sites specified.
        """
        known_values, known_pos = self.__measure_sensor_values(site_values)
        for transformation in self.__symmetry:
            known_pos = transformation(known_pos)
            known_values = np.concatenate((known_values, known_values), axis=0)

        self.__field.fit_model(known_pos, known_values)


    def predict_data(self, field_pos :np.ndarray[float]) -> np.ndarray[float]:
        """
        Predict values of the field at the points specified.

        Args:
            field_pos (np.ndarray[float]): n by d array of n positions of dimension d.

        Returns:
            np.ndarray[float]: n by m array of n values of dimension m.
        """
        return self.__field.predict_values(field_pos)
    

    def calc_keys(self, num_repetitions :int) -> np.ndarray[bool]:
        """
        Calculate a number of potential arrays for the active sensors based off the chances that the sensors fail.

        Args:
            num_repetitions (int): number of keys needed.

        Returns:
            np.ndarray[bool]: n by s array of n keys of dimension s where s is the number of sensors.
        """
        keys = np.full((num_repetitions, self.__num_sensors), True)
        for i, key in enumerate(keys):
            for j in range(len(key)):
                num = np.random.rand()
                if num < self.__sensors[j].get_failure_chance():
                    keys[i, j] = False
        return keys
    

    def get_num_sensors(self):
        """
        Returns:
            _type_: number of sensors.
        """
        return self.__sensors.size






class SymmetryManager():
    """
    Allows assumptions about the symmetry of the temperature field to be built into the predicted temperature field.
    """
    def __init__(self) -> None:
        """
        Initialises the symmetry parameters.
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
            x_pos (np.ndarray[float]): n by 1 array of n 1D points.

        Returns:
            np.ndarray[float]: 2n by 1 array of 2n 1D points.
        """
        axis = np.ones(x_pos.shape)*self.__x_point
        reflected_arr = 2*axis - x_pos
        return np.concatenate((x_pos, reflected_arr), axis=0)


    def reflect_2D_horiz(self, pos :np.ndarray[float]) -> np.ndarray[float]:
        """
        Reflects an array of positions parallel to the x axis. 
        Returns both reflected and original positions.

        Args:
            pos (np.ndarray[float]): n by 2 array of n 2D original points.

        Returns:
            np.ndarray[float]: 2n by 2 array of 2n 2D points.
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
            pos (np.ndarray[float]): n by 2 array of n 2D original points.

        Returns:
            np.ndarray[float]: 2n by 2 array of 2n 2D points.
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
            pos (np.ndarray[float]): n by 2 array of n 2D original points.

        Returns:
            np.ndarray[float]: 2n by 2 array of 2n 2D points.
        """
        m = self.__grad
        reflect_matrix = 1/(1+m**2)*np.array([[1 - m**2, 2*m], [2*m, m**2 - 1]])
        reflected_arr = np.apply_along_axis(reflect_matrix.dot, 0, pos.T)
        return np.concatenate((pos, reflected_arr.T), axis=0)