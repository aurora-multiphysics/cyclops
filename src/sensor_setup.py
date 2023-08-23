from sklearn.linear_model import LinearRegression
from scipy.interpolate import CubicSpline
from sensors import RoundSensor
import numpy as np
import pickle
import os



class ThermocoupleDataReader():
    def __init__(self, file_name) -> None:
        parent_path = os.path.dirname(os.path.dirname(__file__))
        self.__file_path = os.path.join(os.path.sep,parent_path, 'sensors', file_name)


    def generate_sensor_data(self):
        temperatures = []
        voltages = []

        next_temp = 0
        f = open(self.__file_path, "r")
        for i, line in enumerate(f):
            line = line.strip()
            arr_line = line.split(' ')
            if arr_line[0].isdigit():
                if int(arr_line[0]) == next_temp:
                    add_on = 0
                    for volt in arr_line[1:]:
                        if add_on != 10:
                            if volt.replace('.','',1).isdigit():
                                print(next_temp + add_on, volt)
                                temperatures.append(next_temp + add_on)
                                voltages.append(float(volt))
                                add_on += 1
                    next_temp += 10
        return np.array(temperatures), np.array(voltages)


        





if __name__ == '__main__':
    reader = ThermocoupleDataReader('k-type.txt')
    temps, volts = reader.generate_sensor_data()

    temp_to_volts = CubicSpline(temps, volts)
    volt_to_temp = LinearRegression().fit(volts.reshape(-1), temps.reshape(-1))

    def f(temp):
        voltage = temp_to_volts(temp)
        return volt_to_temp.predict(voltage.reshape(-1)) - temp.reshape(-1)

    k_thermcouple = RoundSensor(0.6, f)
    thermo_file = open('thermocouple-k.obj', 'wb')
    pickle.dump(k_thermcouple, thermo_file)
    thermo_file.close()
