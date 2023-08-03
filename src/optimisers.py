from pymoo.algorithms.soo.nonconvex.pso import PSO
from pymoo.algorithms.soo.nonconvex.ga import GA
from src.results_manager import ResultsManager
from pymoo.termination import get_termination
from pymoo.core.problem import Problem
from matplotlib import pyplot as plt
from src.csv_reader import CSVReader
from pymoo.optimize import minimize
import scienceplots
import numpy as np







class SymmetricLossFunction(Problem):
    def __init__(self, num_sensors):
        low_border = [0, -0.0135] * (num_sensors//2)
        high_border = [0.0135, 0.0215] * (num_sensors//2)

        super().__init__(
            n_var=num_sensors, 
            n_obj=1, 
            n_ieq_constr=0, 
            xl=low_border, 
            xu=high_border
        )
        self.__csv_reader = CSVReader('temperature_field.csv')


    def _evaluate(self, swarm_values, out, *args, **kwargs):
        out['F'] = np.apply_along_axis(self.__csv_reader.get_symmetric_loss, 1, swarm_values)



class UniformLossFunction(Problem):
    def __init__(self, num_sensors):
        low_border = [-0.0135, -0.0135] * num_sensors
        high_border = [0.0135, 0.0215] * num_sensors

        super().__init__(
            n_var=num_sensors*2,
            n_obj=1, 
            n_ieq_constr=0, 
            xl=low_border, 
            xu=high_border
        )
        self.__csv_reader = CSVReader('temperature_field.csv')


    def _evaluate(self, swarm_values, out, *args, **kwargs):
        out['F'] = np.apply_along_axis(self.__csv_reader.get_uniform_loss, 1, swarm_values)





def optimise_with_GA(problem):
    algorithm = GA(
        pop_size=50,
        eliminate_duplicates=True)
    termination = get_termination("time", "00:10:00")

    res = minimize(problem,
                algorithm,
                termination,
                seed=5,
                save_history=True,
                verbose=True)
    return res



def optimise_with_PSO(problem):
    algorithm = PSO(
        pop_size=20,
        adaptive = True
    )
    termination = get_termination("time", "00:00:30")

    res = minimize(problem,
                algorithm,
                termination,
                seed=2,
                save_history=True,
                verbose=True)
    return res




def plot_optimsiation(history):
    n_evals = []
    average_loss = []
    min_loss = []

    for algo in history:
        n_evals.append(algo.evaluator.n_eval)
        opt = algo.opt

        min_loss.append(opt.get("F").min())
        average_loss.append(algo.pop.get("F").mean())

    plt.yscale('log')
    plt.plot(n_evals, average_loss, label='average loss')
    plt.plot(n_evals, min_loss, label = 'minimum loss')
    plt.xlabel('Function evaluations')
    plt.ylabel('Function loss')
    plt.legend()
    plt.show()
    plt.close()




def check_results(res, is_symmetric, num_sensors):
    if is_symmetric == True:
        results_manager = ResultsManager('best_symmetric_setups.txt')
    else:
        results_manager = ResultsManager('best_uniform_setups.txt')

    if res.F[0] < results_manager.read_file(num_sensors)[0]:
        print('\nSaving new record...')
        results_manager.write_file(num_sensors, res.F[0], list(res.X))
        results_manager.save_updates()



def show_results(res, is_symmetric):
    plot_optimsiation(res.history)
    csv_reader = CSVReader('temperature_field.csv')

    sensor_positions = []
    for i in range(0, len(res.X), 2):
        sensor_positions.append(csv_reader.find_nearest_pos(res.X[i:i+2]))
    sensor_positions = np.array(sensor_positions)

    print(sensor_positions)
    csv_reader.plot_model(sensor_positions.reshape(-1), symmetric=is_symmetric)
    csv_reader.plot_2D(sensor_positions.reshape(-1), symmetric=is_symmetric)




if __name__ == '__main__':
    plt.style.use('science')
    num_sensors = 3
    symmetric_approach = False

    if symmetric_approach == True:
        res = optimise_with_PSO(SymmetricLossFunction(num_sensors))
        check_results(res, True, num_sensors)
        show_results(res, True)
    else:
        res = optimise_with_PSO(UniformLossFunction(num_sensors))
        check_results(res, False, num_sensors)
        show_results(res, False)




    

