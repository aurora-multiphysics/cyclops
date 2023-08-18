from scipy.interpolate import LinearNDInterpolator
from sensors import Thermocouple
import pandas as pd
import numpy as np
import os



#Monoblock values
X_BOUNDS = (-0.0135, 0.0135)
Y_BOUNDS = (-0.0135, 0.0215)
Z_BOUNDS = (0, 0.012)
MONOBLOCK_RADIUS = 0.006
THERMOCOUPLE_RADIUS = 0.00075
LOSS_LIMIT = 1e6



class CSVReader():
    def __init__(self, csv_name) -> None:
        # Load the file
        parent_path = os.path.dirname(os.path.dirname(__file__))
        full_path = os.path.join(os.path.sep,parent_path,'simulation', csv_name)
        dataframe = pd.read_csv(full_path)
        
        self.__interpolation_pos = self.generate_positions(dataframe)
        self.__interpolation_temps = dataframe['T'].values
        self.__interpolater = LinearNDInterpolator(self.__interpolation_pos, self.__interpolation_temps)
        self.__mean_kernel = self.generate_kernel()
    

    def get_positions(self) -> np.ndarray:
        return self.__interpolation_pos


    def get_temperatures(self) -> np.ndarray:
        return self.__interpolation_temps


    def generate_positions(self, dataframe) -> np.ndarray:
        # Get the position and temperature vectors from the file
        x_values = (dataframe['X'].values)
        y_values = (dataframe['Y'].values)
        return np.concatenate((x_values.reshape(-1, 1), y_values.reshape(-1, 1)), 1)


    def generate_kernel(self, num_x = 5, num_y = 5) -> np.ndarray:
        # Produces the kernel used for calculating the mean temperature in a circle
        x_values = np.linspace(-THERMOCOUPLE_RADIUS, THERMOCOUPLE_RADIUS, num_x)
        y_values = np.linspace(-THERMOCOUPLE_RADIUS, THERMOCOUPLE_RADIUS, num_y)
        pos_in_radius = []
        for x in x_values:
            for y in y_values:
                if x**2 + y**2 <= THERMOCOUPLE_RADIUS**2:
                    pos = np.array([x, y])
                    pos_in_radius.append(pos)
        return np.array(pos_in_radius) 


    def find_temps(self, pos) -> np.ndarray:
        return self.__interpolater(pos)


    def find_mean_temp(self, pos) -> float:
        # Calculates the mean temperature in a circle
        pos_in_radius = self.__mean_kernel + np.ones(self.__mean_kernel.shape)*pos
        temps = self.find_temps(pos_in_radius)
        return np.mean(temps)





class ModelUser():
    def __init__(self, model_type: type, reader: CSVReader) -> None:
        self._comparison_temps = reader.get_temperatures()
        self._comparison_pos = reader.get_positions()
        self._model_type = model_type
        self._reader = reader
        self._sensor = Thermocouple('k-type.csv')
        
    
    def get_model_type(self) -> type:
        return self._model_type


    def find_model_temps(self, pos, model):
        return None

    
    def rearrange_sensor_layout(self, proposed_sensor_layout):
        pass


    def build_trained_model(self, proposed_sensor_layout):
        return None


    def compare_fields(self, model) -> float:
        loss_array = np.square(self._comparison_temps- self.find_model_temps(self._comparison_pos, model))
        return np.sum(loss_array)
    
    
    def find_loss(self, proposed_sensor_layout, repetitions=6000) -> tuple:
        rearranged_layout = self.rearrange_sensor_layout(proposed_sensor_layout)
        sensor_keys, sensor_chances = self.find_sensor_keys_chances(rearranged_layout)
        losses = np.zeros(sensor_keys.shape)

        for i, sensor_key in enumerate(sensor_keys):
            adjusted_layout = self.key_to_layout(rearranged_layout, sensor_key)
            setup_specific_losses = np.zeros(int(np.ceil(sensor_chances[i]*repetitions)))
            for j in range(len(setup_specific_losses)):
                model = self.build_trained_model(adjusted_layout)
                setup_specific_losses[j] = self.compare_fields(model)
            losses[i] = np.mean(setup_specific_losses)
        
        expected_value = np.mean(sensor_chances * losses)
        success_chance = 0.0
        for i, sensor_chance in enumerate(sensor_chances):
            if losses[i] < LOSS_LIMIT:
                success_chance += sensor_chance

        return expected_value, 1-success_chance

    
    def key_to_layout(self, rearranged_sensor_layout, sensor_key) -> np.ndarray:
        new_layout = []
        for i, c in enumerate(sensor_key):
            if c == 'O':
                new_layout.append(rearranged_sensor_layout[i])
        return np.array(new_layout)


    def key_to_lost(self, rearranged_sensor_layout, sensor_key) -> np.ndarray:
        lost_sensors = []
        for i, c in enumerate(sensor_key):
            if c == 'X':
                lost_sensors.append(rearranged_sensor_layout[i])
        return np.array(lost_sensors)


    def find_sensor_keys_chances(self, sensor_layout) -> tuple:
        # Returns a string describing which sensors failed
        num_sensors = len(sensor_layout)
        sensor_keys = ['O'*num_sensors]
        sensor_chances = []

        for i in range(num_sensors):
            setup = ['O']*num_sensors
            setup[i]='X'
            str_setup = ''.join(setup)
            sensor_keys.append(str_setup)
            for j in range(i, num_sensors):
                for k in range(j+1, num_sensors):
                    setup = ['O']*num_sensors
                    setup[i]='X'
                    setup[j]='X'
                    setup[k]='X'
                    str_setup = ''.join(setup)
                    sensor_keys.append(str_setup)
        
        for sensor_key in sensor_keys:
            sensor_chances.append(self.find_chance(sensor_key))

        return np.array(sensor_keys), np.array(sensor_chances)


    def find_chance(self, sensor_key) -> float:
        chance = 1.0
        for letter in sensor_key:
            if letter == 'O':
                chance *= (1-self._sensor.get_failure_chance())
            else:
                chance *= self._sensor.get_failure_chance()
        return chance
    

    def find_train_temps(self, sensor_layout) -> np.ndarray:
        # Return an array containing the temperature at each sensor position
        sensor_temperatures = self._reader.find_temps(sensor_layout)
        sensor_temperatures.reshape(-1)
        for i, temp in enumerate(sensor_temperatures):
            sensor_temperatures[i] = self._sensor.get_linearised_temp(temp)
            sensor_temperatures[i] += self._sensor.get_error()
        return sensor_temperatures
    

    def compare_arrays(self, arr_1, arr_2) -> np.ndarray:
        only_in_2 = []
        for pos in arr_2:
            if np.sum(np.square(pos)) not in np.sum(np.square(arr_1), axis=1):
                only_in_2.append(pos)
        return np.array(only_in_2)


    def find_temps_for_plotting(self, proposed_sensor_layout) -> tuple:
        rearranged_layout = self.rearrange_sensor_layout(proposed_sensor_layout)
        sensor_keys, sensor_chances = self.find_sensor_keys_chances(rearranged_layout)
        losses = np.zeros(sensor_keys.shape)
        sensor_layouts = []
        lost_sensors = []
        model_temps = []

        print('\nDescribing temperature fields...')
        for i, sensor_key in enumerate(sensor_keys):
            adjusted_layout = self.key_to_layout(rearranged_layout, sensor_key)
            model = self.build_trained_model(adjusted_layout)
            losses[i] = self.compare_fields(model)

            sensor_layouts.append(adjusted_layout)
            lost_sensors.append(self.key_to_lost(rearranged_layout, sensor_key))
            model_temps.append(self.find_model_temps(self._comparison_pos, model))
        
        return sensor_layouts, lost_sensors, model_temps, losses, sensor_chances, sensor_keys

    
    


    




class SymmetricManager(ModelUser):
    def __init__(self, model_type, exodus_reader) -> None:
        ModelUser.__init__(self, model_type, exodus_reader)
        
    
    def is_symmetric(self) -> bool:
        return True


    def find_model_temps(self, positions, model):
        return model.get_temp(positions)


    def rearrange_sensor_layout(self, proposed_sensor_layout):
        # Add all of the sensor coordinates that need to be reflected in the x-axis onto the monoblock
        proposed_sensor_layout = proposed_sensor_layout.reshape(-1, 2)
        multiplier = np.array([[-1, 1]]*proposed_sensor_layout.shape[0])
        return np.concatenate((proposed_sensor_layout, multiplier * proposed_sensor_layout), axis=0)


    def build_trained_model(self, adjusted_layout):
        training_temperatures = self.find_train_temps(adjusted_layout)
        return self._model_type(adjusted_layout, training_temperatures)

    



class UniformManager(ModelUser):
    def __init__(self, model_type, exodus_reader):
        ModelUser.__init__(self, model_type, exodus_reader)


    def is_symmetric(self):
        return False

    
    def find_model_temps(self, positions, model):
        y_values = positions[:, 1].reshape(-1, 1)
        return model.get_temp(y_values)


    def rearrange_sensor_layout(self, proposed_sensor_layout):
        return proposed_sensor_layout.reshape(-1, 2)


    def build_trained_model(self, adjusted_layout):
        training_temperatures = self.find_train_temps(adjusted_layout)
        sensor_y_values = adjusted_layout[:,1].reshape(-1, 1)
        return self._model_type(sensor_y_values, training_temperatures)