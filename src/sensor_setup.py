from sensors import Thermocouple, PointSensor
from file_reader import PickleManager
import numpy as np
import os



class ThermocoupleReader():
    """
    Class to read the tabular thermocouple data in the 'sensors' folder.
    This was downloaded from https://srdata.nist.gov/its90/download/download.html.
    """
    def __init__(self, file_name :str) -> None:
        """
        Calculates the path to the file.

        Args:
            file_name (str): the name of the file in the sensors folder.
        """
        parent_path = os.path.dirname(os.path.dirname(__file__))
        self.__file_path = os.path.join(os.path.sep,parent_path, 'sensors', file_name)


    def generate_thermo_data(self) -> tuple[np.ndarray]:
        """
        Reads the file.

        Returns:
            tuple[np.ndarray]: The temperatures and corresponding voltage arrays
        """
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
        return np.array(temperatures).reshape(-1, 1), np.array(voltages).reshape(-1, 1)





if __name__ == '__main__':
    """
    This is run to save the thermocouple data to the sensors folder where it can quickly be read.
    """
    reader = ThermocoupleReader('k-type.txt')
    pickle_manager = PickleManager()

    temperatures, voltages = reader.generate_thermo_data()
    pickle_manager.save_file('sensors', 'k-type-T.obj', temperatures)
    pickle_manager.save_file('sensors', 'k-type-V.obj', voltages)