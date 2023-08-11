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
THERMOCOUPLE_RADIUS = 0.000075




class CSVReader():
    def __init__(self, csv_name):
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


    def generate_positions(self, dataframe):
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


    def find_mean_temp(self, pos):
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


    def build_trained_model(self, proposed_sensor_layout):
        return None


    def compare_fields(self, model):
        loss_array = np.square(self._comparison_temps- self.find_model_temps(self._comparison_pos, model))
        return np.sum(loss_array)
    
    
    def find_loss(self, proposed_sensor_layout, repetitions=20):
        losses = np.zeros(repetitions)
        setups_failed = 0
        for i in range(repetitions):
            model = self.build_trained_model(proposed_sensor_layout)
            if model == None:
                setups_failed += 1
            else:
                losses[i] = self.compare_fields(model)
        return np.mean(losses), setups_failed/repetitions
    

    def find_train_temps(self, sensor_layout):
        # Return an array containing the temperature at each sensor position
        adjusted_sensor_layout = []
        sensor_temperatures = []

        for i, sensor_pos in enumerate(sensor_layout):
            mean_temp = self._reader.find_mean_temp(sensor_pos)
            sensor_temp = self._sensor.get_measured_temp(mean_temp)

            if sensor_temp != None:
                sensor_temperatures.append(sensor_temp)
                adjusted_sensor_layout.append(sensor_pos)
        
        return np.array(adjusted_sensor_layout), np.array(sensor_temperatures)
    

    def compare_arrays(self, arr_1, arr_2):
        only_in_2 = []
        for pos in arr_2:
            if np.sum(np.square(pos)) not in np.sum(np.square(arr_1), axis=1):
                only_in_2.append(pos)
        return np.array(only_in_2)




    
    


    




class SymmetricManager(ModelUser):
    def __init__(self, model_type, exodus_reader):
        ModelUser.__init__(self, model_type, exodus_reader)
        
    
    def is_symmetric(self):
        return True


    def find_model_temps(self, positions, model):
        return model.get_temp(positions)


    def reflect_position(self, proposed_sensor_layout):
        # Add all of the sensor coordinates that need to be reflected in the x-axis onto the monoblock
        proposed_sensor_layout = proposed_sensor_layout.reshape(-1, 2)
        multiplier = np.array([[-1, 1]]*proposed_sensor_layout.shape[0])
        return np.concatenate((proposed_sensor_layout, multiplier * proposed_sensor_layout), axis=0)


    def build_trained_model(self, proposed_sensor_layout):
        symmetric_sensor_layout = self.reflect_position(proposed_sensor_layout)
        adjusted_sensor_layout, training_temperatures = self.find_train_temps(symmetric_sensor_layout)
        if len(training_temperatures) > 3:
            return self._model_type(adjusted_sensor_layout, training_temperatures)
        else:
            return None
    

    def find_temps_for_plotting(self, proposed_sensor_layout):
        symmetric_sensor_layout = self.reflect_position(proposed_sensor_layout)
        adjusted_sensor_layout, training_temperatures = self.find_train_temps(symmetric_sensor_layout)

        if len(training_temperatures) < 3:
            return None, None, None
        model = self._model_type(adjusted_sensor_layout, training_temperatures)

        lost_sensors = self.compare_arrays(adjusted_sensor_layout, symmetric_sensor_layout)
        return self.find_model_temps(self._comparison_pos, model), adjusted_sensor_layout, lost_sensors

    



class UniformManager(ModelUser):
    def __init__(self, model_type, exodus_reader):
        ModelUser.__init__(self, model_type, exodus_reader)


    def is_symmetric(self):
        return False

    
    def find_model_temps(self, positions, model):
        y_values = positions[:, 1].reshape(-1, 1)
        return model.get_temp(y_values)


    def build_trained_model(self, proposed_sensor_layout):
        uniform_sensor_layout = proposed_sensor_layout.reshape(-1, 2)
        adjusted_sensor_layout, training_temperatures = self.find_train_temps(uniform_sensor_layout)
        sensor_y_values = adjusted_sensor_layout[:,1].reshape(-1, 1)
        if len(sensor_y_values) > 3:
            return self._model_type(sensor_y_values, training_temperatures)
        else:
            return None
    

    def find_temps_for_plotting(self, proposed_sensor_layout):
        uniform_sensor_layout = proposed_sensor_layout.reshape(-1, 2)
        adjusted_sensor_layout, training_temperatures = self.find_train_temps(uniform_sensor_layout)

        if len(training_temperatures) < 3:
            return None, None, None
        sensor_y_values = adjusted_sensor_layout[:,1].reshape(-1, 1)
        model = self._model_type(sensor_y_values, training_temperatures)

        lost_sensors = self.compare_arrays(adjusted_sensor_layout, uniform_sensor_layout)
        return self.find_model_temps(self._comparison_pos, model), adjusted_sensor_layout, lost_sensors