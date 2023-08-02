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






# CONSTANTS
HALF_NUM_SENSORS = 5
LOW_BORDER = [0, -0.0135] * HALF_NUM_SENSORS
HIGH_BORDER = [0.0135, 0.0215] * HALF_NUM_SENSORS




class LossFunction(Problem):
    def __init__(self):
        super().__init__(n_var=2*HALF_NUM_SENSORS, n_obj=1, n_ieq_constr=0, xl=LOW_BORDER, xu=HIGH_BORDER)
        self.__csv_reader = CSVReader('temperature_field.csv')


    def _evaluate(self, swarm_values, out, *args, **kwargs):
        out['F'] = np.apply_along_axis(self.__csv_reader.get_loss, 1, swarm_values)





def optimise_with_GA(problem):
    algorithm = GA(
        pop_size=50,
        eliminate_duplicates=True)
    termination = get_termination("time", "00:10:00")

    res = minimize(problem,
                algorithm,
                termination,
                seed=4,
                save_history=True,
                verbose=True)
    return res



def optimise_with_PSO(problem):
    algorithm = PSO(
        pop_size=20,
        adaptive = True
    )
    termination = get_termination("time", "00:00:05")

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








if __name__ == '__main__':
    plt.style.use('science')
    results_manager = ResultsManager()

    print("\nOptimising...")
    res = optimise_with_PSO(LossFunction())
    plot_optimsiation(res.history)
    best_setup = res.X
    csv_reader = CSVReader('temperature_field.csv')

    if res.F[0] < results_manager.read_file(HALF_NUM_SENSORS*2)[0]:
        print('\nSaving new record...')
        results_manager.write_file(HALF_NUM_SENSORS*2, res.F[0], list(res.X))
        results_manager.save_updates()

    print("\nBest setup:")
    sensor_positions = []
    for i in range(0, len(best_setup), 2):
        sensor_positions.append(csv_reader.find_nearest_pos(best_setup[i:i+2]))
    
    sensor_positions = np.array(sensor_positions)
    print(sensor_positions)
    print(csv_reader.get_loss(best_setup))
    csv_reader.plot_model(sensor_positions.reshape(-1))
    csv_reader.plot_2D(sensor_positions.reshape(-1))
