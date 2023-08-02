from matplotlib import pyplot as plt
import scienceplots
import numpy as np
import os


plt.style.use('science')


class ResultsManager():
    def __init__(self):
        script_path = os.path.realpath(__file__)
        parent_path = os.path.dirname(os.path.dirname(script_path))
        self.__setups_path = os.path.join(os.path.sep,parent_path,"results","best_setups.txt")

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


    def write_file(self, write_num, write_result, write_layout):
        index = self.__setup_numbers.index(write_num)
        self.__setup_results[index] = write_result
        self.__setup_layout[index] = write_layout

    
    def read_file(self, read_num):
        index = self.__setup_numbers.index(read_num)
        return self.__setup_results[index], self.__setup_layout[index]


    def save_updates(self):
        f = open(self.__setups_path, "w")
        for i, num in enumerate(self.__setup_numbers):
            line = str(num)+':'+str(self.__setup_results[i])+':'+str(self.__setup_layout[i])
            if num != self.__setup_numbers[-1]:
                line += '\n'
            f.writelines(line)
        f.close()


    def plot_pareto(self):
        plt.scatter(self.__setup_numbers[1:], self.__setup_results[1:], facecolors='none', edgecolors='b')
        plt.xlabel('Number of sensors')
        plt.ylabel('Loss')
        plt.title('Pareto front')
        plt.show()
        plt.close()



if __name__ == '__main__':
    manager = ResultsManager()
    manager.plot_pareto()