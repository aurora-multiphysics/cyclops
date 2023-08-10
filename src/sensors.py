import numpy as np


class Thermocouple():
    def __init__(self):
        self.__failure_chance = 0.1
        self.__range = [-270, 1260]             # -270 to 1260 degrees C
        self.__error = 2.2                      # +/- 2.2 degrees C

    
    def get_measured_temp(self, mean_temp, add_error=True):
        chance = np.random.rand()
        if chance < self.__failure_chance:
            return None
        elif mean_temp > self.__range[1]:
            return None
        elif mean_temp < self.__range[0]:
            return None
        else:
            return mean_temp + self.__error/3 * np.random.normal()





class ThermalCamera():
    pass




class DIC():
    pass