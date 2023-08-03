from src.optimisers import optimise_with_GA, optimise_with_PSO
from src.results_manager import ResultsManager
from pymoo.core.problem import Problem
from src.csv_reader import CSVReader
import numpy as np




class CustomLossFunction(Problem):
    def __init__(self, half_num_sensors, low_border, high_border):
        super().__init__(n_var=2*half_num_sensors, n_obj=1, n_ieq_constr=0, xl=low_border, xu=high_border)
        self.__csv_reader = CSVReader('temperature_field.csv')


    def _evaluate(self, swarm_values, out, *args, **kwargs):
        out['F'] = np.apply_along_axis(self.__csv_reader.get_loss, 1, swarm_values)





if __name__ == '__main__':
    results_manager = ResultsManager()
    sensor_nums = [8, 10, 12, 14, 16, 18, 20, 22, 24]
    sensor_setups = []
    sensor_performance = []

    for num in sensor_nums:
        half_num_sensors = num//2
        low_border = [0, -0.0135] * half_num_sensors
        high_border = [0.0135, 0.0215] * half_num_sensors

        print("\nOptimising...")
        res = optimise_with_PSO(CustomLossFunction(half_num_sensors, low_border, high_border))
        best_setup = res.X
        csv_reader = CSVReader('temperature_field.csv')

        sensor_setups.append(best_setup)
        sensor_performance.append(res.F)
        print(best_setup)
        print(res.F)

        if res.F[0] < results_manager.read_file(num)[0]:
            print('\nSaving new record...')
            results_manager.write_file(num, res.F[0], list(res.X))
            results_manager.save_updates()
    
    print("\n\n")
    print(sensor_setups)
    print(sensor_performance)
    results_manager.plot_pareto()

