from sklearn.linear_model import LinearRegression
from scipy.interpolate import CubicSpline
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
        return temperatures, voltages


    def save_sensor_data(self, temperatures, voltages, file_name):
        parent_path = os.path.dirname(os.path.dirname(__file__))
        full_path = os.path.join(os.path.sep,parent_path,'sensors', file_name)
        
        temp_to_voltage = CubicSpline(temperatures, voltages)
        voltage_to_temp = LinearRegression(voltages, temperatures)
        





if __name__ == '__main__':
    reader = thermocouple_reader('k-type.txt')
    temperatures, voltages = reader.generate_thermo_data()
    reader.write_to_csv(temperatures, voltages, 'k-type.csv')