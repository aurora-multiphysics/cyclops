from pymoo.algorithms.soo.nonconvex.pso import PSO
from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.termination import get_termination
from pymoo.core.problem import Problem
from matplotlib import pyplot as plt
from src.csv_reader import CSVReader
from pymoo.optimize import minimize
import numpy as np





# CONSTANTS





class LossFunction(Problem):
    def __init__(self, half_num_sensors, low_border, high_border):
        super().__init__(n_var=2*half_num_sensors, n_obj=1, n_ieq_constr=0, xl=low_border, xu=high_border)
        self.__csv_reader = CSVReader('temperature_field.csv')


    def _evaluate(self, swarm_values, out, *args, **kwargs):
        out['F'] = np.apply_along_axis(self.__csv_reader.get_loss, 1, swarm_values)





def optimise_with_GA(problem):
    algorithm = GA(
        pop_size=50,
        eliminate_duplicates=True)
    termination = get_termination("time", "00:00:02")

    res = minimize(problem,
                algorithm,
                termination,
                seed=1,
                save_history=True,
                verbose=True)
    return res



def optimise_with_PSO(problem):
    algorithm = PSO(
        pop_size=20,
        adaptive = True
    )
    termination = get_termination("time", "00:10:00")

    res = minimize(problem,
                algorithm,
                termination,
                seed=1,
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

    sensor_nums = [6, 8, 10, 12, 14, 16, 18, 20, 22, 24]
    sensor_setups = []
    sensor_performance = []

    for num in sensor_nums:
        half_num_sensors = num//2
        low_border = [0, -0.0135] * half_num_sensors
        high_border = [0.0135, 0.0215] * half_num_sensors

        print("\nOptimising...")
        res = optimise_with_GA(LossFunction(half_num_sensors, low_border, high_border))
        best_setup = res.X
        csv_reader = CSVReader('temperature_field.csv')

        sensor_setups.append(best_setup)
        sensor_performance.append(res.F)
        print(best_setup)
        print(res.F)
    
    print("\n\n")
    print(sensor_setups)
    print(sensor_performance)

    plt.scatter(sensor_nums, sensor_performance, facecolors='none', edgecolors='b')
    plt.xlabel('Number of sensors')
    plt.ylabel('Loss')
    plt.title('Pareto front')
    plt.show()
    plt.close()