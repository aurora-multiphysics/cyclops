from simulation.exodus_reader import ExodusManager
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
RADIUS = 0.06


class Explorer(nn.Module):
    def __init__(self, num_sensors, f):
        super().__init__()
        positions = torch.distributions.Uniform(low = BORDERS[0], high = BORDERS[1]).sample((num_sensors,))
        self.__positions = nn.Parameter(positions)
        self.__zeros = torch.zeros(len(positions), 1)
        self.f = f       


    def forward(self):
        return self.f(torch.cat((self.__positions, self.__zeros), -1))


    def check_pos(self, pos):
        if pos[0] < BORDERS[0, 0] or pos[0] > BORDERS[1, 0]:
            return False
        if pos[1] < BORDERS[0, 1] or pos[1] > BORDERS[1, 1]:
            return False
        if pos[0] **2 + pos[1] ** 2 < RADIUS ** 2:
            return False
        return True


    def __repr__(self):
        return "f("+str(self.__positions)+") = "+str(self.f(self.__positions))





def training_loop(explorer, optimizer, n=100):
    losses = []
    for i in tqdm(range(n)):
        loss = explorer()
        print(loss)
        #loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        losses.append(float(loss))  
    return losses
























if __name__ == "__main__":
    manager = ExodusManager('monoblock_out11.e')

    explorer = Explorer(1, manager.get_point_temp)
    opt = torch.optim.Adam(explorer.parameters(), lr=0.001)

    print("\n\nOptimising...")
    losses = training_loop(explorer, opt)
    plt.plot(losses)
    plt.show()
    plt.close()

    print("\n\nResult:")
    print(explorer)