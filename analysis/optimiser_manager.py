from matplotlib import pyplot as plt
from tqdm import tqdm
from torch import nn
import torch_optim
import torch



# Potential optimisers to use:
# torch.optim.Adam(explorer.parameters(), lr=0.001)
# torch.optim.SGD(explorer.parameters(), lr=0.001, momentum=0.9)



def f(pos):
    return torch.sum(torch.sin(pos))




class Explorer(nn.Module):
    def __init__(self, num_dim, f):
        super().__init__()
        pos = torch.distributions.Uniform(-10.0, 10.0).sample((num_dim,))
        self.__pos = nn.Parameter(pos)
        self.f = f       


    def forward(self):
        return self.f(self.__pos)


    def __repr__(self):
        return "f("+str(self.__pos)+") = "+str(self.f(self.__pos))





def training_loop(explorer, optimizer, n=1000):
    losses = []
    for i in tqdm(range(n)):
        loss = explorer()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        losses.append(float(loss))  
    return losses





if __name__ == "__main__":
    explorer = Explorer(2, f)
    #opt = torch.optim.Adam(explorer.parameters(), lr=0.001)
    opt = torch_optim.SimulatedAnnealing(explorer)

    print("\n\nOptimising...")
    losses = training_loop(explorer, opt)
    plt.plot(losses)
    plt.show()
    plt.close()

    print("\n\nResult:")
    print(explorer)