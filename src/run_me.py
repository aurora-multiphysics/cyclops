from model_management import SymmetricManager, UniformManager, CSVReader
from face_model import GPModel, RBFModel, CTModel, CSModel
from optimisers import LossFunction, optimise_with_GA
from results_management import ResultsManager
from graph_management import GraphManager
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
csv_reader = CSVReader('side_field.csv')
model_manager = UniformManager(RBFModel, csv_reader)





def show_sensor_layout(layout):
    positions = csv_reader.get_positions()
    true_temperatures = csv_reader.get_temperatures()
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



def optimise_sensor_layout(num_sensors=6, time_limit='00:05:00'):
    # Optimises the sensor placement
    print('\nOptimising...')
    problem = LossFunction(num_sensors, model_manager)
    res = optimise_with_GA(problem, time_limit)

    graph_manager.draw_reliability_pareto(res.F)
    
    print('\nResult:')
    print(res.X)
    print('\nDisplay:')
    show_sensor_layout(res.X)








if __name__ == '__main__':
    #print(model_manager.find_loss(np.array([0, -0.01, 0, 0, 0, 0.01, 0, 0.02])))
    #optimise_sensor_layout()
    show_sensor_layout(np.array([0, -0.01, 0, 0, 0, 0.01, 0, 0.02]))