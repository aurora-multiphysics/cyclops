from face_model import PlaneModel
from matplotlib import pyplot as plt
from matplotlib import cm
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
        x_values = (dataframe['X'].values)
        y_values = (dataframe['Y'].values)
        self.__positions = np.concatenate((x_values.reshape(-1, 1), y_values.reshape(-1, 1)), 1)
        self.__temp_values = (dataframe['T'].values)

        # Make the position to temperature dictionary so temperature reading is O(1)
        self.__pos_to_temp = {}
        for i, x_value in enumerate(x_values):
            hashable_pos = (x_value, y_values[i])
            self.__pos_to_temp[hashable_pos] = self.__temp_values[i]


    def find_nearest_pos(self, pos):
        # Return the nearest position to the pos given
        difference_array = np.square(self.__positions - pos)
        index = np.apply_along_axis(np.sum, 1, difference_array).argmin()
        return self.__positions[index]


    def get_temp(self, pos):
        # Return the temperature at the recorded position nearest pos
        rounded_pos = self.find_nearest_pos(pos)
        rounded_pos = tuple(rounded_pos)
        return self.__pos_to_temp[rounded_pos]


    def plot_3D(self):
        # Plot a smart 3D graph of the temperature at various points of the monoblock
        fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
        surf = ax.plot_trisurf(
            self.__positions[:,0].reshape(-1), 
            self.__positions[:,1].reshape(-1), 
            self.__temp_values, 
            cmap=cm.jet, linewidth=0.1)

        fig.colorbar(surf, shrink=0.5, aspect=5)
        plt.show()
        plt.close()


    def get_loss(self, sensor_positions):
        # Calculate the loss of a configuration of sensor positions
        sensor_temperatures = np.zeros(len(sensor_positions)//2)

        for i in range(0, len(sensor_positions), 2):
            sensor_temperatures[i//2] = self.get_temp(sensor_positions[i:i+2])
        model = PlaneModel(sensor_positions, sensor_temperatures)

        loss = 0
        for pos in self.__positions:
            loss += np.square(self.__pos_to_temp[tuple(pos)] - model.get_temp(pos))
        return loss





if __name__ == "__main__":
    csv_reader = CSVReader('temperature_field.csv')
    print(csv_reader.get_temp([-0.0132273,-0.0092576]))
    csv_reader.plot_3D()
    print(csv_reader.get_loss([-0.0132273,-0.0092576, 0.01, 0.01, -0.01, -0.01]))

