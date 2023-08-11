from sklearn.linear_model import LinearRegression
from scipy.interpolate import CubicSpline
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import os


class Thermocouple():
    def __init__(self, csv_name):
        self.__failure_chance = 0.1             # 1 in 10 fail
        self.__error = 2.2                      # +/- 2.2 degrees C

        parent_path = os.path.dirname(os.path.dirname(__file__))
        file_path = os.path.join(os.path.sep,parent_path, 'simulation', csv_name)
        dataframe = pd.read_csv(file_path)

        temps = dataframe['T'].values
        voltages = dataframe['V'].values

        self.__regressor = LinearRegression()
        self.__regressor.fit(voltages.reshape(-1, 1), temps.reshape(-1, 1))
        self.__extrapolator = np.polynomial.polynomial.Polynomial.fit(temps, voltages, deg=3)
        self.__interpolator = CubicSpline(temps, voltages)

        x = np.linspace(20, 1600, 300)
        new_temps = []
        for temp in x:
            new_temps.append(self.get_linearised_temp(temp))
        plt.plot(x, new_temps)
        #plt.plot(x, x)
        plt.xlabel('True temp')
        plt.ylabel('Linearised temp')
        plt.show()
        plt.close()

    
    def get_linearised_temp(self, start_temp):
        if 0 < start_temp and start_temp < 1370:
            voltage = self.__interpolator(start_temp)
        else:
            voltage = self.__extrapolator(start_temp)
        return self.__regressor.predict(voltage.reshape(-1, 1))[0]
            

    
    def get_measured_temp(self, mean_temp, add_error=True):
        chance = np.random.rand()
        if chance < self.__failure_chance:
            return None
        else:
            return mean_temp + self.__error/3 * np.random.normal()
        

    





class ThermalCamera():
    pass




class DIC():
    pass






if __name__ == '__main__':
    sensor = Thermocouple('k-type.csv')