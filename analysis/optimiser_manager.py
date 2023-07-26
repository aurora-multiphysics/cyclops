from loss_function import LossFunction
from matplotlib import pyplot as plt
from tqdm import tqdm
from torch import nn
import numpy as np
import torch



OFFSET = [1, 2]

def f(pos):
    return torch.sum(torch.sin(pos))



#Monoblock values
BORDERS = torch.Tensor([[-0.0135, -0.0135], [0.0135, 0.0215]])
RADIUS = 0.006


class Explorer(nn.Module):
    def __init__(self, num_sensors, f):
        super().__init__()
        positions = torch.distributions.Uniform(low = BORDERS[0], high = BORDERS[1]).sample((num_sensors,))
        self.__positions = nn.Parameter(positions)
        self.__best = self.__positions.detach().clone()
        self.f = f       


    def forward(self):
        return self.f(self.__positions)

    
    def ensure_correct_pos(self):
        with torch.no_grad():


            for i, sensor_pos in enumerate(self.__positions):
                if self.check_pos(sensor_pos) == False:
                    print("Dodgy sensor position")
                    self.__positions[i] = torch.distributions.Uniform(low = BORDERS[0], high = BORDERS[1]).sample((1,))
            # if self.f(self.__positions) < self.f(self.__best):
            #     self.__best = self.__positions.detach().clone()


    def check_pos(self, pos):
        if pos[0] <= BORDERS[0, 0] or pos[0] >= BORDERS[1, 0]:
            return False
        if pos[1] <= BORDERS[0, 1] or pos[1] >= BORDERS[1, 1]:
            return False
        return True


    def __repr__(self):
        return "f("+str(self.__positions)+") = "+str(self.f(self.__positions))


    def get_best(self):
        return self.__best.detach(), self.f(self.__best.detach())





def training_loop(explorer, optimizer, n = 1000):
    losses = []
    for i in tqdm(range(n)):
        loss = explorer()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        explorer.ensure_correct_pos()
        losses.append(float(loss)) 
    return losses






if __name__ == "__main__":
    loss = LossFunction()

    explorer = Explorer(3, loss.get_loss)
    opt = torch.optim.Adam(explorer.parameters(), lr=0.001)

    print("\n\nOptimising...")
    losses = training_loop(explorer, opt)
    plt.plot(losses)
    plt.show()
    plt.close()

    print("\n\nResult:")
    print(explorer)
    print("\n", explorer.get_best())

    loss.plot_3D_model(explorer.get_best()[0])