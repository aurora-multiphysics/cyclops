import pandas as pd
import numpy as np
import os





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

        # Make the position to temperature dictionary so temperature reading is O(1)
        self._pos_to_temp = {}
        for i, x_value in enumerate(x_values):
            hashable_pos = (x_value, y_values[i])
            self._pos_to_temp[hashable_pos] = self._temp_values[i]
    

    def get_all_positions(self):
        return self._positions


    def get_all_temp_values(self):
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


    def find_train_temps(self, sensor_layout):
        num_sensors = len(sensor_layout)
        sensor_temperatures = np.zeros(num_sensors)

        for i in range(0, num_sensors):
            rounded_pos = self.find_nearest_pos(sensor_layout[i])
            sensor_temperatures[i] = self.get_temp(rounded_pos)
        return sensor_temperatures




class ModelUser():
    def get_loss(self, proposed_sensor_layout):
        model = self.get_trained_model(proposed_sensor_layout)
        return self.compare_fields(model)

    
    def get_trained_model(self, proposed_sensor_layout):
        return None


    def get_model_temp(self, pos, model):
        return None

    
    def get_all_model_temperatures(self, sensor_layout):
        model = self.get_trained_model(sensor_layout)
        model_temperatures = np.zeros(len(self._positions))

        for i in range(len(self._positions)):
            model_temperatures[i] = self.get_model_temp(self._positions[i], model)
        return model_temperatures


    def compare_fields(self, model):
        loss = 0
        for pos in self._positions:
            loss += np.square(self._pos_to_temp[tuple(pos)] - self.get_model_temp(pos, model))
        return loss





class SymmetricManager(CSVReader, ModelUser):
    def __init__(self, csv_name, model_type):
        super().__init__(csv_name)
        self._default_model_type = model_type
    
    
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
        training_temperatures = self.find_train_temps(symmetric_sensor_layout)
        return self._default_model_type(symmetric_sensor_layout, training_temperatures)

    

class UniformManager(CSVReader, ModelUser):
    def __init__(self, csv_name, model_type):
        super().__init__(csv_name)
        self._default_model_type = model_type


    def is_symmetric(self):
        return False

    
    def get_model_temp(self, pos, model):
        return model.get_temp(pos[1])


    def get_trained_model(self, proposed_sensor_layout):
        uniform_sensor_layout = proposed_sensor_layout.reshape(-1, 2)
        training_temperatures = self.find_train_temps(uniform_sensor_layout)
        sensor_y_values = uniform_sensor_layout[:,1].reshape(-1, 1)
        return self._default_model_type(sensor_y_values, training_temperatures)

    









