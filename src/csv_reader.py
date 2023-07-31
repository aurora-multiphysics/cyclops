from matplotlib import pyplot as plt
from src.face_model import GPModel
from matplotlib import tri as tri
from matplotlib import cm
import scienceplots
import pandas as pd
import numpy as np
import os





RADIUS = 0.006



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


    def setup_model(self, sensor_positions):
        # Returns the model from the sensor positions
        sensor_temperatures = np.zeros(len(sensor_positions)//2)

        for i in range(0, len(sensor_positions), 2):
            sensor_temperatures[i//2] = self.get_temp(sensor_positions[i:i+2])
        model = GPModel(sensor_positions, sensor_temperatures)
        return model


    def get_loss(self, sensor_positions):
        # Calculate the loss of a configuration of sensor positions
        model = self.setup_model(sensor_positions)

        loss = 0
        for pos in self.__positions:
            loss += np.square(self.__pos_to_temp[tuple(pos)] - model.get_temp(pos))
        return loss


    def plot_model(self, sensor_positions):
        # Setup the model
        model = self.setup_model(sensor_positions)

        # Plot the real temperatures
        fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
        surf = ax.plot_trisurf(
            self.__positions[:,0].reshape(-1), 
            self.__positions[:,1].reshape(-1), 
            self.__temp_values, 
            cmap=cm.jet, linewidth=0.1
        )
        fig.colorbar(surf, shrink=0.5, aspect=5)
        
        # Plot the model's temperatures
        predicted_temperatures = []
        for pos in self.__positions:
            predicted_temperatures.append(model.get_temp(pos))
        
        surf = ax.plot_trisurf(
            self.__positions[:,0].reshape(-1), 
            self.__positions[:,1].reshape(-1), 
            predicted_temperatures
        )
        
        plt.show()
        plt.close()


    def plot_2D(self, sensor_positions):
        model = self.setup_model(sensor_positions)
        plt.style.use('science')

        # Plot the real temperatures
        fig, (ax_1, ax_2, ax_3) = plt.subplots(1,3)
        cp_1 = ax_1.tricontourf(
            self.__positions[:,0].reshape(-1), 
            self.__positions[:,1].reshape(-1), 
            self.__temp_values, 
            cmap=cm.jet, levels = 30
        )
        ax_1.set_title('Simulation temperature field')
        ax_1.set_xlabel('x (m)')
        ax_1.set_ylabel('y (m)')

        # Calculate the model's temperatures
        predicted_temperatures = []
        for pos in self.__positions:
            predicted_temperatures.append(model.get_temp(pos))
        
        # Plot the model's temperatures
        cp_2 = ax_2.tricontourf(
            self.__positions[:,0].reshape(-1), 
            self.__positions[:,1].reshape(-1), 
            predicted_temperatures, 
            cmap=cm.jet, levels = 30
        )
        ax_2.set_title('Sensor data GP temperature field')
        ax_2.set_xlabel('x (m)')
        ax_2.set_ylabel('y (m)')

        c_bar = fig.colorbar(cp_2, ax=[ax_1, ax_2])

        # Plot the sensor positions
        sensor_x = []
        sensor_y = []
        for i in range(len(sensor_positions)):
            if i%2 == 0:
                sensor_x.append(sensor_positions[i])
            else:
                sensor_y.append(sensor_positions[i])
        ax_2.scatter(
            sensor_x, 
            sensor_y,
            s=20,
            color='black'
        )

        # Plot the grid for the monoblock
        triang = tri.Triangulation(self.__positions[:, 0], self.__positions[:, 1])
        triang.set_mask(np.hypot(
            self.__positions[:, 0][triang.triangles].mean(axis=1),
            self.__positions[:, 1][triang.triangles].mean(axis=1)) 
        < RADIUS)
        cp_3 = ax_3.triplot(triang)
        ax_3.set_title('Monoblock setup')
        ax_3.scatter(
            sensor_x, 
            sensor_y,
            s=20,
            color='black'
        )


        plt.show()
        plt.close()





if __name__ == "__main__":
    csv_reader = CSVReader('temperature_field.csv')



    best_sensor_positions = np.array([
        [-0.0097759,  0.0202931],
        [ 0.0107069,  0.0202931],
        [-0.0060517,  0.0130517],
        [ 0.012569,   0.0166724],
        [ 0.0088448,  0.0021897],
        [-0.012569,   0.0106379],
        [-0.0023276,  0.0058103],
        [-0.012569,   0.0033966],
        [ 0.0041897,  0.0106379],
        [ 0.0116379,  0.0082241]
    ]).reshape(-1)
    csv_reader.plot_2D(best_sensor_positions)

