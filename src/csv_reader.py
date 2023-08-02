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
        # Return the nearest potential sensor position to the pos given
        difference_array = np.square(self.__positions - pos)
        index = np.apply_along_axis(np.sum, 1, difference_array).argmin()
        return self.__positions[index]


    def get_temp(self, rounded_pos):
        # Return the temperature at the recorded position nearest pos
        hashable_pos = tuple(rounded_pos)
        return self.__pos_to_temp[hashable_pos]


    def setup_model(self, symmetric_sensor_layout):
        # Returns the model from the sensor positions
        num_sensors = len(symmetric_sensor_layout)
        sensor_temperatures = np.zeros(num_sensors)

        for i in range(0, num_sensors):
            rounded_pos = self.find_nearest_pos(symmetric_sensor_layout[i])
            sensor_temperatures[i] = self.get_temp(rounded_pos)

        model = GPModel(symmetric_sensor_layout, sensor_temperatures)
        return model


    def reflect_position(self, proposed_sensor_layout):
        # Add all of the sensor coordinates that need to be reflected in the x-axis onto the monoblock
        proposed_sensor_layout = proposed_sensor_layout.reshape(-1, 2)
        multiplier = np.array([[-1, 1]]*proposed_sensor_layout.shape[0])
        return np.concatenate((proposed_sensor_layout, multiplier * proposed_sensor_layout), axis=0)


    def get_loss(self, proposed_sensor_layout):
        # Calculate the loss of a configuration of sensor positions
        symmetric_sensor_layout = self.reflect_position(proposed_sensor_layout)
        model = self.setup_model(symmetric_sensor_layout)

        loss = 0
        for pos in self.__positions:
            loss += np.square(self.__pos_to_temp[tuple(pos)] - model.get_temp(pos))
        return loss




    # Below are the visualisation functions. I seriously though about creating an extra class to 
    # hold them to improve the simplicity of the architecture, but this would have been super
    # unnecessary and over the top so I didn't.




    def plot_model(self, proposed_sensor_layout):
        # Setup the model
        symmetric_sensor_layout = self.reflect_position(proposed_sensor_layout)
        model = self.setup_model(symmetric_sensor_layout)

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
        [ 0.012569,   0.0058103],
        [ 0.0088448,  0.0202931],
        [ 0.0041897,  0.0118448],
        [ 0.0079138,  0.0046034],
        [ 0.0088448, -0.0074655]
    ]).reshape(-1)


    csv_reader.plot_model(best_sensor_positions)
    csv_reader.plot_2D(best_sensor_positions)
    print(csv_reader.get_loss(best_sensor_positions))


