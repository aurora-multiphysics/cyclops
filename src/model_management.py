from sensors import Thermocouple
from tqdm import tqdm
import pandas as pd
import numpy as np
import os


THERMOCOUPLE_RADIUS = 0.0025


class CSVReader():
    def __init__(self, csv_name):
        # Load the file
        parent_path = os.path.dirname(os.path.dirname(__file__))
        full_path = os.path.join(os.path.sep,parent_path,'simulation', csv_name)
        dataframe = pd.read_csv(full_path)

        # Get the position and temperature vectors from the file
        x_values = (dataframe['X'].values)
        y_values = (dataframe['Y'].values)
        self._positions = np.concatenate((x_values.reshape(-1, 1), y_values.reshape(-1, 1)), 1)
        self._temp_values = (dataframe['T'].values)

        # Make the position to temperature matrix so temperature reading is O(1)
        self._pos_to_temp = {}
        for i, x_value in enumerate(x_values):
            hashable_pos = (x_value, y_values[i])
            self._pos_to_temp[hashable_pos] = self._temp_values[i]

        # Create the mean_temp dictionary
        self._pos_to_mean_temp = {}
        print('\nCalculating mean temperature values...')
        for i, x_value1 in enumerate(tqdm(x_values)):
            hashable_pos = (x_value1, y_values[i])
            temp_in_radius = []
            for j, x_value2 in enumerate(x_values):
                if np.sqrt((x_value2 - x_value1)**2 + (y_values[j] - y_values[i])**2) < THERMOCOUPLE_RADIUS**2:
                    temp_in_radius.append(self._pos_to_temp[(x_value2, y_values[j])])
            self._pos_to_mean_temp[hashable_pos] = np.mean(np.array(temp_in_radius))
    

    def get_positions(self):
        return self._positions


    def get_temp_values(self):
        return self._temp_values


    def find_nearest_pos(self, pos):
        # Return the nearest potential sensor position to the pos given
        difference_array = np.square(self._positions - pos)
        index = np.apply_along_axis(np.sum, 1, difference_array).argmin()
        return self._positions[index]


    def get_temp(self, rounded_pos):
        # Return the temperature at the recorded position nearest pos
        hashable_pos = tuple(rounded_pos)
        return self._pos_to_temp[hashable_pos]
    

    def get_mean_temp(self, rounded_pos):
        # Return the mean at the recorded position nearest pos
        hashable_pos = tuple(rounded_pos)
        return self._pos_to_mean_temp[hashable_pos]


    def find_train_temps(self, sensor_layout):
        # Return an array containing the temperature at each sensor position
        adjusted_sensor_layout = []
        sensor_temperatures = []
        sensor = Thermocouple()

        for i in range(0, len(sensor_layout)):
            rounded_pos = self.find_nearest_pos(sensor_layout[i])
            mean_temp = self.get_mean_temp(rounded_pos)
            sensor_temp = sensor.get_measured_temp(mean_temp)
            
            if sensor_temp != None:
                sensor_temperatures.append(sensor_temp)
                adjusted_sensor_layout.append(rounded_pos)

        return np.array(adjusted_sensor_layout), np.array(sensor_temperatures)
    

    def find_lost_sensors(self, sensor_arr_1, sensor_arr_2):
        lost_sensors = []
        for pos in sensor_arr_1:
            rounded_pos = self.find_nearest_pos(rounded_pos)
            if rounded_pos not in sensor_arr_2:
                lost_sensors.append(rounded_pos)
        return np.array(lost_sensors)




class ModelUser():
    def __init__(self, model_type):
        self._default_model_type = model_type

    
    def get_model_type(self):
        return self._default_model_type

    
    def get_trained_model(self, proposed_sensor_layout):
        return None


    def get_model_temp(self, pos, model):
        return None


    def compare_fields(self, model):
        loss = 0
        for pos in self._positions:
            loss += np.square(self._pos_to_temp[tuple(pos)] - self.get_model_temp(pos, model))
        return loss


    def get_loss(self, proposed_sensor_layout):
        model = self.get_trained_model(proposed_sensor_layout)
        loss= self.compare_fields(model)
        return loss, 1.0
    

    def get_model_temperatures(self, model, positions):
        model_temperatures = np.zeros(len(positions))
        for i in range(len(positions)):
            model_temperatures[i] = self.get_model_temp(positions[i], model)
        return model_temperatures
    
    


    




class SymmetricManager(CSVReader, ModelUser):
    def __init__(self, csv_name, model_type):
        CSVReader.__init__(self, csv_name)
        ModelUser.__init__(self, model_type)
        
    
    def is_symmetric(self):
        return True


    def get_model_temp(self, pos, model):
        return model.get_temp(pos)


    def reflect_position(self, proposed_sensor_layout):
        # Add all of the sensor coordinates that need to be reflected in the x-axis onto the monoblock
        proposed_sensor_layout = proposed_sensor_layout.reshape(-1, 2)
        multiplier = np.array([[-1, 1]]*proposed_sensor_layout.shape[0])
        return np.concatenate((proposed_sensor_layout, multiplier * proposed_sensor_layout), axis=0)


    def get_trained_model(self, proposed_sensor_layout):
        symmetric_sensor_layout = self.reflect_position(proposed_sensor_layout)
        adjusted_sensor_layout, training_temperatures = self.find_train_temps(symmetric_sensor_layout)
        return self._default_model_type(adjusted_sensor_layout, training_temperatures)
    

    def find_temp_for_plotting(self, positions, layout):
        symmetric_sensor_layout = self.reflect_position(layout)
        adjusted_sensor_layout, training_temperatures = self.find_train_temps(symmetric_sensor_layout)
        model = self._default_model_type(adjusted_sensor_layout, training_temperatures)
        return self.get_model_temperatures(model, positions), adjusted_sensor_layout, self.find_lost_sensors(symmetric_sensor_layout, adjusted_sensor_layout)

    



class UniformManager(CSVReader, ModelUser):
    def __init__(self, csv_name, model_type):
        CSVReader.__init__(self, csv_name)
        ModelUser.__init__(self, model_type)


    def is_symmetric(self):
        return False

    
    def get_model_temp(self, pos, model):
        return model.get_temp(pos[1])


    def get_trained_model(self, proposed_sensor_layout):
        uniform_sensor_layout = proposed_sensor_layout.reshape(-1, 2)
        adjusted_sensor_layout, training_temperatures = self.find_train_temps(uniform_sensor_layout)
        sensor_y_values = adjusted_sensor_layout[:,1].reshape(-1, 1)
        return self._default_model_type(sensor_y_values, training_temperatures)
    

    def find_temp_for_plotting(self, positions, layout):
        uniform_sensor_layout = layout.reshape(-1, 2)
        adjusted_sensor_layout, training_temperatures = self.find_train_temps(uniform_sensor_layout)

        sensor_y_values = adjusted_sensor_layout[:,1].reshape(-1, 1)
        model = self._default_model_type(sensor_y_values, training_temperatures)
        return self.get_model_temperatures(model, positions), adjusted_sensor_layout, self.find_lost_sensors(uniform_sensor_layout, adjusted_sensor_layout)
