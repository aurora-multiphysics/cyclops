from matplotlib import pyplot as plt
import pandas as pd
import os



class ThermocoupleReader():
    def __init__(self, file_name) -> None:
        parent_path = os.path.dirname(os.path.dirname(__file__))
        self.__file_path = os.path.join(os.path.sep,parent_path, 'sensors', file_name)


    def generate_thermo_data(self):
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


    def write_to_csv(self, temperatures, voltages, csv_name):
        data = {
            'T': temperatures, 
            'V': voltages
        }
        dataframe = pd.DataFrame(data)
        print('\n', dataframe)

        parent_path = os.path.dirname(os.path.dirname(__file__))
        full_path = os.path.join(os.path.sep,parent_path,'sensors', csv_name)
        dataframe.to_csv(full_path, index=False)





if __name__ == '__main__':
    reader = ThermocoupleReader('k-type.txt')
    temperatures, voltages = reader.generate_thermo_data()
    reader.write_to_csv(temperatures, voltages, 'k-type.csv')