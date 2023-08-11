from model_management import SymmetricManager, UniformManager
from face_model import GPModel, RBFModel, CTModel, CSModel
from optimisers import LossFunction, optimise_with_GA
from results_management import ResultsManager
from graph_management import GraphManager
from exodus_reader import ExodusReader
import numpy as np



MODEL_TO_STRING = {
    GPModel:'GP',
    RBFModel:'RBF',
    CTModel:'CT',
    CSModel:'CS'
}

STRING_TO_MODEL = {
    'GP':GPModel,
    'RBF':RBFModel,
    'CT':CTModel,
    'CS':CSModel
}

graph_manager = GraphManager()
results_manager = ResultsManager('best_setups.txt')
exodus_reader = ExodusReader('monoblock_out.e', 's')
model_manager = UniformManager(CSModel, exodus_reader)





def show_sensor_layout(layout):
    positions = exodus_reader.get_positions()
    true_temperatures = exodus_reader.get_temperatures()
    model_temperatures, new_layout, lost_sensors = model_manager.find_temps_for_plotting(layout)
    
    graph_manager.draw_double_3D_temp_field(
        positions, 
        true_temperatures, 
        model_temperatures
    )
    graph_manager.draw_side_comparison_pane(
        positions, 
        new_layout, 
        true_temperatures, 
        model_temperatures,
        lost_sensors
    )
    print('\nLoss', model_manager.find_loss(layout))



def optimise_sensor_layout(model_manager, num_sensors=10, time_limit='00:10:00'):
    # Optimises the sensor placement
    print('\nOptimising...')
    problem = LossFunction(num_sensors, model_manager)
    res = optimise_with_GA(problem, time_limit)

    graph_manager.draw_reliability_pareto(res.F)
    
    print('\nResult:')
    print(res.X)
    print('\nDisplay:')














if __name__ == '__main__':
    #optimise_sensor_layout(uniform_manager, num_sensors=4, time_limit='00:03:00')
    show_sensor_layout(np.array([0.011, 0.0058103, 0.0088448, 0.0102931, 0.0041897, 0.0118448, 0.0079138, 0.0046034, 0.0088448, -0.0074655]))