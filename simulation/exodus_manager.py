from matplotlib import pyplot as plt
from matplotlib import cm
from tqdm import tqdm
import pandas as pd
import torch
import os


#Monoblock values
X_BOUNDS = (-0.0135, 0.0135)
Y_BOUNDS = (-0.0135, 0.0215)
Z_BOUNDS = (0, 0.012)
RADIUS = 0.006



class ExodusManager():
    def __init__(self, relative_path_to_csv):
        # Load the file
        absolute_path = os.path.dirname(__file__)
        full_path = os.path.join(absolute_path, relative_path_to_csv)
        dataframe = pd.read_csv(full_path)

        # Get the position tensors from the file
        x1 = torch.Tensor(dataframe['X1'].values).reshape(-1, 1)
        y1 = torch.Tensor(dataframe['Y1'].values).reshape(-1, 1)
        x2 = torch.Tensor(dataframe['X2'].values).reshape(-1, 1)
        y2 = torch.Tensor(dataframe['Y2'].values).reshape(-1, 1)
        pos2 = torch.cat((x2, y2), 1)

        # Defining position attributes
        self.__pos1 = torch.cat((x1, y1), 1)
        self.__diff = pos2[0] - self.__pos1[0]
        self.__unique_x = torch.unique(x1)
        self.__num_x = len(self.__unique_x)
        self.__unique_y = torch.unique(y1)
        self.__zeros = torch.zeros(len(self.__unique_x))


        # Get the temperature tensors from the file
        self.__temp1 = torch.Tensor(dataframe['T1'].values).reshape(-1, 1)
        self.__temp3 = torch.Tensor(dataframe['T3'].values).reshape(-1, 1)
        self.__temp4 = torch.Tensor(dataframe['T4'].values).reshape(-1, 1)

    
    def get_temp(self, pos):
        # Get the rounded position
        x_index = torch.argmin(torch.max(pos[0] - self.__unique_x, self.__zeros)) - 1
        y_index = torch.argmin(torch.max(pos[1] - self.__unique_y, self.__zeros)) - 1
        rounded_pos = torch.Tensor([self.__unique_x[x_index], self.__unique_y[y_index]])
        
        # Get the necessary temperatures
        pos_index = x_index * self.__num_x + y_index
        temp_1 = self.__temp1[pos_index]
        relative_temp_3 = self.__temp3[pos_index] - temp_1
        relative_temp_4 = self.__temp4[pos_index] - temp_1

        # Calculate the temperature
        temp = temp_1 + torch.dot(torch.divide(torch.Tensor([relative_temp_3, relative_temp_4]), self.__diff), pos - rounded_pos)
        return temp


    def check_pos(self, pos):
        if pos[0] < X_BOUNDS[0] or pos[0] > X_BOUNDS[1]:
            return False
        if pos[1] < Y_BOUNDS[0] or pos[1] > Y_BOUNDS[1]:
            return False
        if pos[0] ** 2 + pos[1] ** 2 < RADIUS ** 2:
            return False
        return True




    def plot_3D(self):
        # Plot a smart 3D graph of the temperature at various points of the monoblock
        fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
        surf = ax.plot_trisurf(self.__pos1[:,0], self.__pos1[:,1], self.__temp1.reshape(-1), cmap=cm.jet, linewidth=0.1)
        fig.colorbar(surf, shrink=0.5, aspect=5)
        plt.show()
        plt.close()



    


        



if __name__ == "__main__":
    manager = ExodusManager('temperature_field.csv')
    #manager.plot_3D()
    print(manager.get_temp(torch.Tensor([0.01, 0.015])))

