from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.algorithms.soo.nonconvex.pso import PSO
from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.termination import get_termination
from pymoo.operators.crossover.sbx import SBX
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.operators.mutation.pm import PM
from pymoo.optimize import minimize



def moo_GA(problem, time_limit):
    algorithm = NSGA2(
        pop_size=40,
        n_offsprings=10,
        sampling=FloatRandomSampling(),
        crossover=SBX(prob=0.9, eta=15),
        mutation=PM(eta=20),
        eliminate_duplicates=True
    )
    termination = get_termination("time", time_limit)
    
    res = minimize(problem,
                algorithm,
                termination,
                seed=1,
                save_history=True,
                verbose=True)
    return res




def soo_PSO(problem, time_limit):
    algorithm = PSO(
        pop_size=30,
        adaptive=True
    )
    termination = get_termination("time", time_limit)
    
    res = minimize(problem,
                algorithm,
                termination,
                seed=1,
                save_history=True,
                verbose=True)
    return res




def soo_GA(problem, time_limit):
    algorithm = GA(
        pop_size=50,
        eliminate_duplicates=True
    )
    termination = get_termination("time", time_limit)
    
    res = minimize(problem,
                algorithm,
                termination,
                seed=2,
                save_history=True,
                verbose=True)
    return res