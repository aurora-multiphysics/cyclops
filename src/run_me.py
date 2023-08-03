from src.csv_reader import SymmetricReader, UniformReader
from src.face_model import GPModel, RBFModel
from src.graph_manager import GraphManager
import numpy as np



def show_sensor_layout(layout, reader, graph_manager):
    # Plots 3D graph + comparison pane
    positions = reader.get_all_positions()
    true_temperatures = reader.get_all_temp_values()
    model_temperatures = reader.get_all_model_temperatures(layout)
    
    if reader.is_symmetric():
        layout = reader.reflect_position(layout)

    graph_manager.draw_double_3D_temp_field(
        positions, 
        true_temperatures, 
        model_temperatures
    )
    graph_manager.draw_comparison_pane(
        positions, 
        layout.reshape(-1, 2), 
        true_temperatures, 
        model_temperatures
    )








if __name__ == '__main__':
    graph_manager = GraphManager()
    symmetric_reader = SymmetricReader('temperature_field.csv', GPModel)
    uniform_reader = UniformReader('temperature_field.csv', RBFModel)

    layout = np.array([0.012569, 0.0058103, 0.0088448, 0.0202931, 0.0041897, 0.0118448, 0.0079138, 0.0046034, 0.0088448, -0.0074655])
    show_sensor_layout(layout, uniform_reader, graph_manager)