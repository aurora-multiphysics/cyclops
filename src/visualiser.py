from results_manager import ResultsManager
from csv_reader import CSVReader
import numpy as np


def show_setup(num_sensors):
    results_manager = ResultsManager()
    csv_reader = CSVReader('temperature_field.csv')
    loss, layout = results_manager.read_file(num_sensors)

    sensor_positions = np.zeros(num_sensors)
    for i in range(0, len(layout), 2):
        rounded_pos = csv_reader.find_nearest_pos(layout[i:i+2])
        sensor_positions[i] = rounded_pos[0]
        sensor_positions[i+1] = rounded_pos[1]
   
    sensor_positions = np.array(sensor_positions)
    results_manager.plot_pareto() 
    csv_reader.plot_model(sensor_positions.reshape(-1))
    csv_reader.plot_2D(sensor_positions.reshape(-1))



if __name__=='__main__':
    show_setup(10)