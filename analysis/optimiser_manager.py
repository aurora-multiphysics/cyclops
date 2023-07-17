from simulation.exodus_reader import ExodusReader
from simulation.field_manager import FaceManager
from analysis.loss_function import Loss
from matplotlib import pyplot as plt
from tqdm import tqdm
from torch import nn
import torch



def f(pos):
    return torch.sum(torch.sin(pos))



BORDERS = torch.Tensor([[-0.012, 0.012], [-0.011, 0.02]])




class Explorer(nn.Module):
    def __init__(self, num_sensors, f):
        super().__init__()
        pos = torch.distributions.Uniform(low = BORDERS[:0], high = BORDERS[:1]).sample((num_sensors,2))
        self.__pos = nn.Parameter(pos)
        self.f = f       


    def forward(self):
        return self.f(self.__pos)


    def __repr__(self):
        return "f("+str(self.__pos)+") = "+str(self.f(self.__pos))





def training_loop(explorer, optimizer, n=10000):
    losses = []
    for i in tqdm(range(n)):
        loss = explorer()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        losses.append(float(loss))  
    return losses





if __name__ == "__main__":
    reader = ExodusReader("monoblock_out.e")
    face_x, face_y, face_T = reader.front_only()
    managerF = FaceManager(face_x, face_y, face_T, 500)
    loss = Loss(managerF)

    explorer = Explorer(3, loss.loss_function)
    opt = torch.optim.Adam(explorer.parameters(), lr=0.001)

    print("\n\nOptimising...")
    losses = training_loop(explorer, opt)
    plt.plot(losses)
    plt.show()
    plt.close()

    print("\n\nResult:")
    print(explorer)