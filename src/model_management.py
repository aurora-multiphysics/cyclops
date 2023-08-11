from exodus_reader import ExodusReader
from sensors import Thermocouple
import numpy as np




class ModelUser():
    def __init__(self, model_type: type, reader: ExodusReader) -> None:
        self._comparison_temps = reader.get_temperatures()
        self._comparison_pos = reader.get_positions()
        self._model_type = model_type
        self._reader = reader
        
    
    def get_model_type(self) -> type:
        return self._model_type


    def find_model_temp(self, pos, model):
        return None


    def build_trained_model(self, proposed_sensor_layout):
        return None


    def compare_fields(self, model):
        loss = 0
        for pos in self._positions:
            loss += np.square(self._reader.get_temperatures(pos)[0] - self.find_model_temp(pos, model))
        return loss
    
    
    def find_loss(self, proposed_sensor_layout):
        model = self.build_trained_model(proposed_sensor_layout)
        loss = self.compare_fields(model)
        return loss, 1.0
    

    def find_model_temperatures(self, model, positions):
        model_temperatures = np.zeros(len(positions))
        for i in range(len(positions)):
            model_temperatures[i] = self.find_model_temp(positions[i], model)
        return model_temperatures


    def find_train_temps(self, sensor_layout):
        # Return an array containing the temperature at each sensor position
        adjusted_sensor_layout = []
        sensor_temperatures = []
        sensor = Thermocouple()

        for i, sensor_pos in enumerate(sensor_layout):
            mean_temp = self._reader.find_mean_temp(sensor_pos)
            sensor_temp = sensor.get_measured_temp(mean_temp)
            
            if sensor_temp != None:
                sensor_temperatures.append(sensor_temp)
                adjusted_sensor_layout.append(sensor_pos)

        return np.array(adjusted_sensor_layout), np.array(sensor_temperatures)
    

    def compare_arrays(self, arr_1, arr_2):
        only_in_2 = []
        for pos in arr_2:
            if pos not in arr_1:
                only_in_2.append(pos)
        return np.array(only_in_2)




    
    


    




class SymmetricManager(ModelUser):
    def __init__(self, model_type, exodus_reader):
        ModelUser.__init__(self, model_type, exodus_reader)
        
    
    def is_symmetric(self):
        return True


    def find_model_temp(self, pos, model):
        return model.get_temp(pos)


    def reflect_position(self, proposed_sensor_layout):
        # Add all of the sensor coordinates that need to be reflected in the x-axis onto the monoblock
        proposed_sensor_layout = proposed_sensor_layout.reshape(-1, 2)
        multiplier = np.array([[-1, 1]]*proposed_sensor_layout.shape[0])
        return np.concatenate((proposed_sensor_layout, multiplier * proposed_sensor_layout), axis=0)


    def build_trained_model(self, proposed_sensor_layout):
        symmetric_sensor_layout = self.reflect_position(proposed_sensor_layout)
        adjusted_sensor_layout, training_temperatures = self.find_train_temps(symmetric_sensor_layout)
        return self._model_type(adjusted_sensor_layout, training_temperatures)
    

    def find_temps_for_plotting(self, proposed_sensor_layout):
        symmetric_sensor_layout = self.reflect_position(proposed_sensor_layout)
        adjusted_sensor_layout, training_temperatures = self.find_train_temps(symmetric_sensor_layout)
        model = self._model_type(adjusted_sensor_layout, training_temperatures)

        lost_sensors = self.compare_arrays(adjusted_sensor_layout, symmetric_sensor_layout)
        return self.find_model_temperatures(model, self._comparison_pos), adjusted_sensor_layout, lost_sensors

    



class UniformManager(ModelUser):
    def __init__(self, model_type, exodus_reader):
        ModelUser.__init__(self, model_type, exodus_reader)


    def is_symmetric(self):
        return False

    
    def find_model_temp(self, pos, model):
        return model.get_temp(pos[1])


    def build_trained_model(self, proposed_sensor_layout):
        uniform_sensor_layout = proposed_sensor_layout.reshape(-1, 2)
        adjusted_sensor_layout, training_temperatures = self.find_train_temps(uniform_sensor_layout)
        sensor_y_values = adjusted_sensor_layout[:,1].reshape(-1, 1)
        return self._model_type(sensor_y_values, training_temperatures)
    

    def find_temps_for_plotting(self, proposed_sensor_layout):
        uniform_sensor_layout = proposed_sensor_layout.reshape(-1, 2)
        adjusted_sensor_layout, training_temperatures = self.find_train_temps(uniform_sensor_layout)
        sensor_y_values = adjusted_sensor_layout[:,1].reshape(-1, 1)
        model = self._model_type(sensor_y_values, training_temperatures)

        lost_sensors = self.compare_arrays(adjusted_sensor_layout, uniform_sensor_layout)
        return self.find_model_temperatures(model, self._comparison_pos), adjusted_sensor_layout, lost_sensors