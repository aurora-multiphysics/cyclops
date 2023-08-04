import numpy as np
import os



class ResultsManager():
    def __init__(self, file_name):
        parent_path = os.path.dirname(os.path.dirname(__file__))
        self.__setups_path = os.path.join(os.path.sep,parent_path, 'results', file_name)

        # Read the file
        self.__setup_numbers = []
        self.__setup_results = []
        self.__setup_layout = []

        f = open(self.__setups_path, "r")
        for line in f:
            num, result, layout = line.split(':')
            self.__setup_numbers.append(int(num))
            self.__setup_results.append(float(result))
            cropped_layout = layout[1:-2]
            layout_list = cropped_layout.split(',')
            self.__setup_layout.append([float(i) for i in layout_list])
        f.close()

    
    def get_nums(self):
        return self.__setup_numbers


    def write_file(self, write_num, write_result, write_layout):
        # Update the best result & setup for a certain number of sensors
        index = self.__setup_numbers.index(write_num)
        self.__setup_results[index] = write_result
        self.__setup_layout[index] = write_layout

    
    def read_file(self, read_num):
        # Return the best result & setup for a certain number of sensors
        index = self.__setup_numbers.index(read_num)
        return self.__setup_results[index], self.__setup_layout[index]


    def save_updates(self):
        # Save any updates into the file by rewriting to it
        f = open(self.__setups_path, "w")
        for i, num in enumerate(self.__setup_numbers):
            line = str(num)+':'+str(self.__setup_results[i])+':'+str(self.__setup_layout[i])
            if num != self.__setup_numbers[-1]:
                line += '\n'
            f.writelines(line)
        f.close()