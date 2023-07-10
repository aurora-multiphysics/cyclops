from matplotlib import pyplot as plt
import numpy as np




def f(pos):
    return np.sum(np.sin(pos))




class ParticleSwarm():
    def __init__(self, num_dim, borders, num_particles=6, w=0.9, c1=0.2, c2=0.1):
        #Attributes
        self.__swarm = np.random.uniform(borders[0], borders[1], (num_particles, num_dim))
        self.__swarm_v = np.random.uniform(-1, 1, (num_particles, num_dim))
        self.__p_best = np.copy(self.__swarm)
        self.__g_best = self.__p_best[np.argmin(f(self.__p_best))]
        self.__borders = borders
        self.__num_dim = num_dim

        #Hyperparameters
        self.__w = w
        self.__c1 = c1
        self.__c2 = c2



    def run(self, f, repetitions):
        for i in range(repetitions):
            for j in range(len(self.__swarm)):
                #Updates particle positions
                self.__swarm_v[j] = (self.__w*self.__swarm_v[j] + 
                                    np.random.rand(self.__num_dim)*self.__c1*(self.__p_best[j] - self.__swarm[j]) + 
                                    np.random.rand(self.__num_dim)*self.__c2*(self.__g_best - self.__swarm[j]))
                self.__swarm[j] += self.__swarm_v[j]

                #Checks the particle is in the correct range
                for x in self.__swarm[j]:
                    if x < self.__borders[0] or x > self.__borders[1]:
                        x = np.random.uniform(self.__borders[0], self.__borders[1])

                #Updates swarm parameters
                value = f(self.__swarm[j])
                if value <= f(self.__p_best[j]):
                    self.__p_best[j] = self.__swarm[j]
                if value <= f(self.__g_best):
                    self.__g_best = self.__swarm[j]
        return self.__g_best
    



class SimulatedAnnealing():
    def __init__(self, num_dim, borders, T=10.0, final_T = 1.0, step_size=0.9):
        # Attributes
        self.__pos = np.random.uniform(borders[0], borders[1], num_dim)
        self.__g_best = np.copy(self.__pos)
        self.__borders = borders

        # Parameters for optimisation
        self.__temp = T
        self.__end_temp = final_T
        self.__step = step_size

        self.values = []


    def run(self, f, repetitions):
        self.__alpha = (self.__temp - self.__end_temp)/repetitions
        while self.__temp > self.__end_temp:
            # Generates candidate neighbour
            candidate_pos = self.__pos + np.random.uniform(-1, 1)*self.__step

            # Checks the candidate is in the correct region
            for x in self.__pos:
                if x < self.__borders[0] or x > self.__borders[1]:
                    x = np.random.uniform(self.__borders[0], self.__borders[1])
        
            # Is the energy high enough for a transfer?
            diff = f(candidate_pos) - f(self.__pos)
            if np.exp(- diff/self.__temp) >= np.random.rand():
                self.__pos = candidate_pos
            if f(candidate_pos) < f(self.__g_best):
                self.__g_best = candidate_pos
            # Update the temperature
            self.__temp -= self.__alpha
            self.values.append(f(self.__pos))
        return self.__g_best
    




class GeneticAlgorithm():
    def __init__(self, num_dim, borders, population_size=60, beta=0.7, r_mut = 0.5):
        #Population size should be converted to a multiple of 4
        self.__size = (population_size * 4)//4
        self.__population = np.random.uniform(borders[0], borders[1], (self.__size, num_dim))
        self.__num_chromosones = len(self.__population[0])
        self.__global_best = np.copy(self.__population[0])
        self.__borders = borders
        
        #Learning rates
        self.__beta = beta
        self.__r_mut = r_mut


    def select_parents(self, fitness):
        parents = np.zeros((self.__size//2, self.__num_chromosones))
        worst = np.max(fitness)
        for i in range(self.__size//2):
            selection_i = np.argmin(fitness)
            parents[i] = self.__population[selection_i]
            fitness[selection_i] += worst
        return parents
    

    def crossover(self, parent_1, parent_2):
        c1, c2 = np.copy(parent_1), np.copy(parent_2)
        c1 *= self.__beta
        c2 *= self.__beta
        cross_i = np.random.randint(0, self.__num_chromosones)
        c1[cross_i:] = parent_2[cross_i:] * (1 - self.__beta)
        c2[:cross_i] = parent_1[:cross_i] * (1 - self.__beta)
        return c1, c2
    

    def mate_parents(self, parents):
        children = np.zeros((self.__size//2, self.__num_chromosones))
        for i in range(0, self.__size//2, 2):
            children[i], children[i + 1] = self.crossover(
                parents[np.random.randint(self.__num_chromosones)], 
                parents[np.random.randint(self.__num_chromosones)]
                )
        return children
    

    def mutate(self):
        chance = np.random.rand(self.__size, self.__num_chromosones)
        for y in range(self.__size):
            for x in range(self.__num_chromosones):
                if chance[y, x] < self.__r_mut:
                    self.__population[y, x] = np.random.uniform(self.__borders[0], self.__borders[1])

    

    def run(self, f, repetitions):
        for i in range(repetitions):
            fitness = np.apply_along_axis(f, 1, self.__population)
            parents = self.select_parents(fitness)
            children = self.mate_parents(parents)

            #Find best performing sample from generation
            if f(parents[0]) < f(self.__global_best):
                self.__global_best = parents[0] 

            #Proudce new generation
            self.__population = np.concatenate((parents, children), axis=0)
            self.mutate()
        return self.__global_best




if __name__ == "__main__":
    sa = SimulatedAnnealing(2, (-10, 10))
    print("\n", f(sa.run(f, 1000)))
    plt.plot(sa.values)
    plt.show()
    plt.close()
