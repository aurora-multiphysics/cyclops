import torch




class SimulatedAnnealing(torch.optim.Optimizer):
    def __init__(self, params, lr=0.01, T=10.0, step_size=1.0, alpha = 0.9):
        super(SimulatedAnnealing, self).__init__(params)
        self.__temp = T
        self.__step = step_size
        self.__alpha = alpha
  
    def step(self):
        pass
        #TODO