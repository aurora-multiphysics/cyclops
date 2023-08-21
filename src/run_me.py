from model_management import SymmetricManager, UniformManager, CSVReader
from face_model import GPModel, RBFModel, CTModel, CSModel, LSModel
from optimisers import LossFunction, optimise_with_GA
from graph_management import GraphManager, PDFManager
from results_management import ResultsManager
import numpy as np



MODEL_TO_STRING = {
    GPModel:'GP',
    RBFModel:'RBF',
    CTModel:'CT',
    CSModel:'CS',
    LSModel:'LS'
}

STRING_TO_MODEL = {
    'GP':GPModel,
    'RBF':RBFModel,
    'CT':CTModel,
    'CS':CSModel,
    'LS':LSModel
}

graph_manager = GraphManager()
results_manager = ResultsManager('best_setups.txt')
csv_reader = CSVReader('side_field.csv')




def optimise_sensor_layout(model_manager, num_sensors=5, time_limit='00:00:10'):
    # Optimises the sensor placement
    print('\nOptimising...')
    problem = LossFunction(num_sensors, model_manager)
    res = optimise_with_GA(problem, time_limit)
    #graph_manager.draw_reliability_pareto(res.F)
    
    print('\nResult:')
    print(res.X)
    results_manager.write_file(
        MODEL_TO_STRING[model_manager.get_model_type()], 
        res.X.tolist())
    results_manager.save_updates()
    return res


def save_setup(model_manager, layout, name):
    positions = csv_reader.get_positions()
    true_temperatures = csv_reader.get_temperatures()

    sensor_layouts, lost_sensors, model_temperatures, losses, chances, sensor_keys = model_manager.find_temps_for_plotting(layout)
    graph_manager.create_pdf(
        positions, 
        sensor_layouts, 
        true_temperatures, 
        model_temperatures, 
        lost_sensors, 
        's', 
        losses, 
        chances,
        MODEL_TO_STRING[model_manager.get_model_type()],
        sensor_keys,
        name
    )


def pareto_search(time_of_search='08:00:00'):
    for num_s in range(6, 8):
        # model_manager = UniformManager(RBFModel, csv_reader)
        # res = optimise_sensor_layout(model_manager, num_sensors=num_s, time_limit=time_of_search)
        # graph_manager.save_reliability_pareto(res.F, str(num_s)+'RBF.png')
        # for i, setup in enumerate(res.X):
        #     save_setup(model_manager, setup, str(num_s)+'RBF'+str(i)+'.pdf')

        model_manager = UniformManager(GPModel, csv_reader)
        res = optimise_sensor_layout(model_manager, num_sensors=num_s, time_limit=time_of_search)
        graph_manager.save_reliability_pareto(res.F, str(num_s)+'GP.png')
        for i, setup in enumerate(res.X):
            save_setup(model_manager, setup, str(num_s)+'GP'+str(i)+'.pdf')




if __name__ == '__main__':
    # Note that for GP we need num_sensors >= 5 
    model_manager = UniformManager(RBFModel, csv_reader)
    res = optimise_sensor_layout(model_manager, num_sensors=6)
    save_setup(model_manager, res.X[0], 'test.pdf')
    #pareto_search()