from model_management import SymmetricManager, UniformManager, CSVReader
from face_model import GPModel, RBFModel, CTModel, CSModel
from optimisers import LossFunction, optimise_with_GA
from graph_management import GraphManager, PDFManager
from results_management import ResultsManager
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




def optimise_sensor_layout(num_sensors=5, time_limit='00:00:30'):
    # Optimises the sensor placement
    print('\nOptimising...')
    problem = LossFunction(num_sensors, model_manager)
    res = optimise_with_GA(problem, time_limit)

    graph_manager.draw_reliability_pareto(res.F)
    
    print('\nResult:')
    print(res.X)
    show_setup(res.X[0])


def show_setup(layout):
    positions = csv_reader.get_positions()
    true_temperatures = csv_reader.get_temperatures()

    sensor_layouts, lost_sensors, model_temperatures, losses, chances = model_manager.find_temps_for_plotting(layout)
    graph_manager.create_pdf(
        positions, 
        sensor_layouts, 
        true_temperatures, 
        model_temperatures, 
        lost_sensors, 
        's', 
        losses, 
        chances,
        MODEL_TO_STRING[model_manager.get_model_type()]
    )



if __name__ == '__main__':
    # Note that for GP we need num_sensors >= 5 
    #print(model_manager.find_loss(np.array([0.001, -0.01, 0.001, 0, 0.001, 0.01, 0.001, 0.02])))
    #show_setup(np.array([0.001, -0.01, 0.001, 0, 0.001, 0.01, 0.001, 0.02]))

    optimise_sensor_layout()