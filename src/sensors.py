import numpy as np


class thermocouple():
    def __init__(self, failure_chance, true_temp, temps_in_radius):
        self.__failure_chance = failure_chance
        self.__true_temp = true_temp
        self.__measured_temp = np.mean(temps_in_radius)

    
    def get_measured_temp(self):
        chance = np.random.rand()
        if chance < self.__failure_chance:
            return None
        else:
            return self.__measured_temp





class thermal_camera():
    pass



