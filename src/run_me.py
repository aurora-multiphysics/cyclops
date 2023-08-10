from model_management import SymmetricManager, UniformManager
from face_model import GPModel, RBFModel, CTModel, CSModel
from optimisers import LossFunction, optimise_with_GA
from results_management import results_manager
from graph_management import graph_manager
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




def show_sensor_layout(layout, model_manager):
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
    graph_manager.draw_side_comparison_pane(
        positions, 
        layout, 
        true_temperatures, 
        model_temperatures
    )



def optimise_sensor_layout(model_manager, num_sensors=10, time_limit='00:10:00'):
    # Optimises the sensor placement
    print('\nOptimising...')
    problem = LossFunction(num_sensors, model_manager)
    res = optimise_with_GA(problem, time_limit)

    graph_manager.draw_reliability_pareto(res.F)
    
    print('\nResult:')
    print(res.X)
    print('\nDisplay:')
    save_results(res.X, model_manager)
    show_results(res.X, model_manager)



def show_results(res_x, model_manager):
    if len(res_x) > 5:
        res_x = res_x[:5]
    for setup in res_x:
        show_sensor_layout(setup, model_manager)



def save_results(res_x, model_manager):
    ask_save = ''
    while ask_save not in ('Y', 'N'):
        ask_save = input('Do you want to save the sensor layouts [Y/N]? ')
    if ask_save == 'N':
        return None
    ask_ID = results_manager.get_IDs()[0]
    while ask_ID in results_manager.get_IDs():
        ask_ID = input('Enter ID to save model: ')

    if model_manager.is_symmetric():
        id = 'S'+ask_ID
    else:
        id = 'U'+ask_ID

    results_manager.write_file(
        id, 
        MODEL_TO_STRING[model_manager.get_model_type],
        res_x 
    )
    results_manager.save_updates()



def find_pareto(model_manager, time_limit='00:10:00', sensor_nums=[14, 16]):
    setups = []
    for num in sensor_nums:
        problem = LossFunction(num, model_manager)
        print('\nOptimising...')
        res = optimise_with_GA(problem, time_limit)
        print('\nResult:')
        print(res.X)
        setups.append(res.X)
    print('\nSetups:\n',setups)



def show_old_setups(old_ID):
    model, setups = results_manager.read_file(old_ID)
    if old_ID[0] == 'U':
        temp_model_manager = UniformManager('side_field.csv', STRING_TO_MODEL[model])
    else:
        temp_model_manager = SymmetricManager('side_field.csv', STRING_TO_MODEL[model])
    setups = np.array(setups)
    show_results(setups, temp_model_manager)








if __name__ == '__main__':
    symmetric_manager = SymmetricManager('side_field.csv', GPModel)
    uniform_manager = UniformManager('side_field.csv', CSModel)

    #optimise_sensor_layout(uniform_manager, num_sensors=4, time_limit='00:03:00')
    show_old_setups('S5-1')