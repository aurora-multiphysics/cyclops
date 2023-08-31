import numpy as np



class Sensor():
    def __init__(
            self, 
            noise_dev :float, 
            offset_function :callable, 
            failure_chance :float, 
            value_range :np.ndarray[float],
            measurement_sites :np.ndarray[float]
        ) -> None:
        self._noise_dev = noise_dev
        self._offset_function = offset_function
        self._failure_chance = failure_chance
        self._range = value_range

        self._measurement_sites = measurement_sites
        self._ground_truth = None

    
    def get_failure_chance(self):
        return self._failure_chance


    def get_sites(self):
        return self._measurement_sites

    
    def set_values(self, values: np.ndarray[float]):
        self._ground_truth = np.mean(values, axis=0)

    
    def get_value(self):
        self._squash_to_range()
        noise_array = np.random.normal(0, self._noise_dev, size=self._ground_truth.size)
        return self._ground_truth + noise_array + self._offset_function(self._ground_truth)


    def _squash_to_range(self):
        for i, value in enumerate(self._ground_truth):
            if value < self._range[0]:
                self._ground_truth[i] = self._range[0]
            elif self._ground_truth > self._range[1]:
                self._ground_truth[i] = self._range[1]



class PointSensor1D(Sensor):
    def __init__(
            self, 
            noise_dev: float, 
            offset_function: callable, 
            failure_chance: float, 
            value_range: np.ndarray[float]
        ) -> None:
        measurement_sites = np.array([[0]])
        super().__init__(noise_dev, offset_function, failure_chance, value_range, measurement_sites)



class PointSensor2D(Sensor):
    def __init__(
            self, 
            noise_dev: float, 
            offset_function: callable, 
            failure_chance: float, 
            value_range: np.ndarray[float]
        ) -> None:
        measurement_sites = np.array([[0, 0]])
        super().__init__(noise_dev, offset_function, failure_chance, value_range, measurement_sites)




class RoundSensor1D(Sensor):
    def __init__(
            self, 
            noise_dev: float, 
            offset_function: callable, 
            failure_chance: float, 
            value_range: np.ndarray[float],
            radius :float
        ) -> None:
        measurement_sites = np.array([[0], [0], [0], [-radius], [radius]])
        super().__init__(noise_dev, offset_function, failure_chance, value_range, measurement_sites)



class RoundSensor2D(Sensor):
    def __init__(
            self, 
            noise_dev: float, 
            offset_function: callable, 
            failure_chance: float, 
            value_range: np.ndarray[float],
            radius :float
        ) -> None:
        measurement_sites = np.array([[0, 0], [0, radius], [0, -radius], [-radius, 0], [radius, 0]])
        super().__init__(noise_dev, offset_function, failure_chance, value_range, measurement_sites)

    


class MultiSensor1D(Sensor):
    def __init__(
            self, 
            noise_dev: float, 
            offset_function: callable, 
            failure_chance: float, 
            value_range: np.ndarray[float],
            grid :np.ndarray[float]
        ) -> None:
        super().__init__(noise_dev, offset_function, failure_chance, value_range, grid)

    
    def set_values(self, values: np.ndarray[float]):
        self._ground_truth = values



class MultiSensor2D(Sensor):
    def __init__(
            self, 
            noise_dev: float, 
            offset_function: callable, 
            failure_chance: float, 
            value_range: np.ndarray[float],
            grid :np.ndarray[float]
        ) -> None:
        super().__init__(noise_dev, offset_function, failure_chance, value_range, grid)

    
    def set_values(self, values: np.ndarray[float]):
        self._ground_truth = values





class Thermocouple1D(RoundSensor1D):
    def __init__(self, noise_dev: float, offset_function: callable, failure_chance: float, value_range: np.ndarray[float], radius: float) -> None:
        super().__init__(noise_dev, offset_function, failure_chance, value_range, radius)




class Thermocouple2D(RoundSensor2D):
    def __init__(self, noise_dev: float, offset_function: callable, failure_chance: float, value_range: np.ndarray[float], radius: float) -> None:
        super().__init__(noise_dev, offset_function, failure_chance, value_range, radius)