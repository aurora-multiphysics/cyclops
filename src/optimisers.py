from pymoo.algorithms.soo.nonconvex.pso import PSO
from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.visualization.scatter import Scatter
from pymoo.termination import get_termination
from pymoo.core.problem import Problem
from pymoo.optimize import minimize
from src.csv_reader import CSVReader
import numpy as np





# CONSTANTS
NUM_SENSORS = 10
LOW_BORDER = [-0.0135, -0.0135] * NUM_SENSORS
HIGH_BORDER = [0.0135, 0.0215] * NUM_SENSORS




class TestFunction(Problem):
    def __init__(self):
        super().__init__(n_var=2*NUM_SENSORS, n_obj=1, n_ieq_constr=0, xl=LOW_BORDER, xu=HIGH_BORDER)


    def _evaluate(self, positions, out, *args, **kwargs):
        out['F'] = np.sum(np.sin(positions), axis=1)





class LossFunction(Problem):
    def __init__(self):
        super().__init__(n_var=2*NUM_SENSORS, n_obj=1, n_ieq_constr=0, xl=LOW_BORDER, xu=HIGH_BORDER)
        self.__csv_reader = CSVReader('temperature_field.csv')


    def _evaluate(self, swarm_values, out, *args, **kwargs):
        out['F'] = np.apply_along_axis(self.__csv_reader.get_loss, 1, swarm_values)





def optimise_with_GA(problem):
    algorithm = GA(
        pop_size=100,
        eliminate_duplicates=True)
    termination = get_termination("time", "00:10:00")

    res = minimize(problem,
                algorithm,
                termination,
                seed=1,
                verbose=True)
    return res.X



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
                verbose=True)
    return res.X







if __name__ == '__main__':
    print("\nOptimising...")
    best_setup = optimise_with_PSO(LossFunction())
    csv_reader = CSVReader('temperature_field.csv')
    


    print("\nBest setup:")
    sensor_positions = []
    for i in range(0, len(best_setup), 2):
        sensor_positions.append(csv_reader.find_nearest_pos(best_setup[i:i+2]))

    sensor_positions = np.array(sensor_positions)
    print(sensor_positions)
    print(csv_reader.get_loss(best_setup))
    csv_reader.plot_model(best_setup.reshape(-1))
