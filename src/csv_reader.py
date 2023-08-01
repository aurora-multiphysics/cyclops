from src.face_model import GPModel, IDWModel
from matplotlib import pyplot as plt
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


    def setup_model(self, sensor_positions):
        # Returns the model from the sensor positions
        sensor_temperatures = np.zeros(len(sensor_positions))

        for i in range(0, len(sensor_positions)):
            sensor_temperatures[i] = self.get_temp(sensor_positions[i])

        model = GPModel(sensor_positions, sensor_temperatures)
        return model


    def reflect_position(self, sensor_coordinates):
        # Add all of the sensor coordinates that need to be reflected in the x-axis onto the monoblock
        sensor_coordinates = sensor_coordinates.reshape(-1, 2)
        multiplier = np.array([[-1, 1]]*sensor_coordinates.shape[0])
        return np.concatenate((sensor_coordinates, multiplier * sensor_coordinates), axis=0)


    def get_loss(self, sensor_positions):
        # Calculate the loss of a configuration of sensor positions
        sensor_coordinates = self.reflect_position(sensor_positions)
        model = self.setup_model(sensor_coordinates)

        loss = 0
        for pos in self.__positions:
            loss += np.square(self.__pos_to_temp[tuple(pos)] - model.get_temp(pos))
        return loss


    def plot_model(self, sensor_positions):
        # Setup the model
        sensor_positions = self.reflect_position(sensor_positions)
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
        sensor_positions = self.reflect_position(sensor_positions)
        model = self.setup_model(sensor_positions)

        # Plot the real temperatures
        fig, (ax_1, ax_2, ax_3) = plt.subplots(1,3, figsize=(18, 7))
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
            sensor_x.append(sensor_positions[i, 0])
            sensor_y.append(sensor_positions[i, 1])
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

        #Plot the differences
        differences = np.zeros(len(self.__temp_values))
        for i in range(len(differences)):
            differences[i] = abs(self.__temp_values[i] - predicted_temperatures[i])

        fig, ax = plt.subplots(layout='constrained', figsize=(5, 6))
        cp = ax.tricontourf(
            self.__positions[:,0].reshape(-1), 
            self.__positions[:,1].reshape(-1), 
            differences, 
            cmap=cm.Blues, levels = 30
        )
        ax.set_title('Absolute differences')
        ax.set_xlabel('x (m)')
        ax.set_ylabel('y (m)')

        cbar = fig.colorbar(cp)
        plt.show()
        plt.close()




if __name__ == "__main__":
    csv_reader = CSVReader('temperature_field.csv')
    plt.style.use('science')

    best_sensor_positions = np.array([
        [ 0.0079138,  0.0046034],
        [ 0.012569,   0.0058103],
        [ 0.0088448,  0.0082241],
        [ 0.0116379,  0.0202931],
        [ 0.012569,  -0.0086724]
    ]).reshape(-1)

    # best_sensor_positions = np.array([
        # [ 0.012569,   0.0082241],
        # [ 0.0079138, -0.0002241],
        # [ 0.0013966,  0.0070172],
        # [ 0.0116379,  0.0033966],
        # [ 0.012569,   0.0058103],
        # [ 0.0004655, -0.0074655],
        # [ 0.0051207,  0.0082241],
        # [ 0.0023276,  0.0202931],
        # [ 0.0004655,  0.0070172],
        # [ 0.0079138,  0.0190862],
        # [ 0.0069828, -0.0062586],
        # [ 0.0013966,  0.0130517],
        # [ 0.0116379,  0.0058103],
        # [ 0.0060517,  0.0046034],
        # [ 0.0116379, -0.0062586],
        # [ 0.0088448,  0.0082241],
        # [ 0.012569,  -0.0062586],
        # [ 0.0060517,  0.0178793],
        # [ 0.0088448,  0.0118448],
        # [ 0.0060517, -0.0002241],
        # [ 0.0060517,  0.0202931],
        # [ 0.0107069,  0.0202931]
    # ]).reshape(-1)

    csv_reader.plot_model(best_sensor_positions)
    csv_reader.plot_2D(best_sensor_positions)
    print(csv_reader.get_loss(best_sensor_positions))


