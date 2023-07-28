from matplotlib import pyplot as plt
from matplotlib import cm
from tqdm import tqdm
import pandas as pd
import numpy as np
import os





class CSVReader():
    def __init__(self, relative_path_to_csv):
        # Load the file
        absolute_path = os.path.dirname(__file__)
        full_path = os.path.join(absolute_path, relative_path_to_csv)
        dataframe = pd.read_csv(full_path)

        # Get the position and temperature vectors from the file
        self.__x_values = (dataframe['X'].values)
        self.__y_values = (dataframe['Y'].values)
        self.__temp_values = (dataframe['T'].values)
        positions = np.concatenate((self.__x_values.reshape(-1, 1), self.__y_values.reshape(-1, 1)), axis=1)

        # Make the position to temperature dictionary so temperature reading is O(1)
        self.__pos_to_temp = {}
        for i, pos in enumerate(positions):
            hashable_pos = tuple(pos)
            self.__pos_to_temp[hashable_pos] = self.__temp_values[i]

    
    def get_temp(self, pos):
        # Return the temperature at a specific x, y position (pos is a tuple)
        return self.__pos_to_temp[pos]


    def get_potential_positions(self):
        # Returns two arrays of the potential sensor positions
        return np.unique(self.__x_values), np.unique(self.__y_values)


    def plot_3D(self):
        # Plot a smart 3D graph of the temperature at various points of the monoblock
        fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
        surf = ax.plot_trisurf(self.__x_values, self.__y_values, self.__temp_values, cmap=cm.jet, linewidth=0.1)
        fig.colorbar(surf, shrink=0.5, aspect=5)
        plt.show()
        plt.close()


    def get_loss(self, sensor_pos):
        pass






if __name__ == "__main__":
    csv_reader = CSVReader('temperature_field.csv')
    print(csv_reader.get_temp((0.0132273,-0.008904)))
    csv_reader.plot_3D()

