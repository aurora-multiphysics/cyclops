from face_model import GPModel, RBFModel, CTModel, CTRBFModel, CSModel
from model_management import SymmetricManager, UniformManager
from optimisers import LossFunction, optimise_with_GA
from results_manager import ResultsManager
from graph_manager import GraphManager
import numpy as np



def show_sensor_layout(layout, model_manager, graph_manager):
    # Plots 3D graph + comparison pane
    positions = model_manager.get_positions()
    true_temperatures = model_manager.get_temp_values()
    model_temperatures = model_manager.get_model_temperatures(layout, positions)
    print('\nLoss:', str(model_manager.get_loss(layout)))
    
    if model_manager.is_symmetric():
        layout = model_manager.reflect_position(layout)
    else:
        layout = layout.reshape(-1, 2)
    for i, pos in enumerate(layout):
        layout[i] = model_manager.find_nearest_pos(pos)

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



def show_pareto(graph_manager, is_symmetric=True):
    if is_symmetric == True:
        results_manager = ResultsManager('best_symmetric_setups.txt')
    else:
        results_manager = ResultsManager('best_uniform_setups.txt')
    
    numbers = results_manager.get_nums()
    results = []
    for num in numbers:
        results.append(results_manager.read_file(num)[0])
    graph_manager.draw_pareto(numbers, results)



def check_results(res, is_symmetric, num_sensors, model_name):
    if is_symmetric == True:
        results_manager = ResultsManager('best_symmetric_non_ideal_setups.txt')
    else:
        results_manager = ResultsManager('best_uniform_non_ideal_setups.txt')

    if res.F[0] < results_manager.read_file(num_sensors)[0]:
        print('\nSaving new record...')
        results_manager.write_file(num_sensors, res.F[0], list(res.X), model_name)
        results_manager.save_updates()



def optimise_sensor_layout(model_manager, graph_manager, num_sensors=10, time_limit='00:10:00'):
    # Optimises the sensor placement
    print('\nOptimising...')
    problem = LossFunction(num_sensors, model_manager)
    res = optimise_with_GA(problem, time_limit)

    model_type_to_name = {
        GPModel:'GP',
        RBFModel:'RBF',
        CTModel:'CT',
        CTRBFModel:'CTRBF',
        CSModel:'CS'
    }
    # check_results(
    #     res, 
    #     model_manager.is_symmetric(), 
    #     num_sensors,
    #     model_type_to_name[model_manager.get_model_type()]
    # )
    # graph_manager.draw_optimisation(res.history)
    graph_manager.draw_reliability_pareto(res.F)
    print('\nResult:')
    print(res.X)
    for layout in res.X:
        show_sensor_layout(
            layout, 
            model_manager, 
            graph_manager
        )



def show_best(graph_manager, model_manager, num):
    if model_manager.is_symmetric() == True:
        results_manager = ResultsManager('best_symmetric_non_ideal_setups.txt')
    else:
        results_manager = ResultsManager('best_uniform_non_ideal_setups.txt')
    
    loss, layout, model_name = results_manager.read_file(num)
    show_sensor_layout(
        np.array(layout), 
        model_manager, 
        graph_manager
    )
    



def find_pareto(model_manager, time_limit='00:10:00', sensor_nums=[14, 16]):
    setups = []
    for num in sensor_nums:
        problem = LossFunction(num, model_manager)
        print('\nOptimising...')
        res = optimise_with_GA(problem, time_limit)

        model_type_to_name = {
        GPModel:'GP',
        RBFModel:'RBF',
        CTModel:'CT',
        CTRBFModel:'CTRBF',
        CSModel:'CS'
        }
        check_results(
            res, 
            model_manager.is_symmetric(), 
            num,
            model_type_to_name[model_manager.get_model_type()]
        )
        print('\nResult:')
        print(res.X)
        setups.append(res.X)
    print('\nSetups:\n',setups)








if __name__ == '__main__':
    graph_manager = GraphManager()
    # Note that the uniform manager can never manage the CTModel
    symmetric_manager = SymmetricManager('temperature_field.csv', GPModel)
    uniform_manager = UniformManager('temperature_field.csv', CSModel)

    #layout = np.array([0.012569, 0.0058103, 0.0088448, 0.0202931, 0.0041897, 0.0118448, 0.0079138, 0.0046034, 0.0088448, -0.0074655])
    #show_sensor_layout(layout, symmetric_manager, graph_manager)
    #show_pareto(graph_manager, False)
    #show_best(graph_manager, uniform_manager, 3)
    optimise_sensor_layout(uniform_manager, graph_manager, num_sensors=4, time_limit='00:05:00')
    #find_pareto(symmetric_manager)