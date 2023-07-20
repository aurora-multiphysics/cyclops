from matplotlib import pyplot as plt
from matplotlib import cm
import chigger
import torch
import os



#Monoblock values
X_BOUNDS = (-0.0135, 0.0135)
Y_BOUNDS = (-0.0135, 0.0215)
Z_BOUNDS = (0, 0.012)



class ExodusManager():
    def __init__(self, relative_path):
        # Load the file
        absolute_path = os.path.dirname(__file__)
        full_path = os.path.join(absolute_path, relative_path)
        reader = chigger.exodus.ExodusReader(full_path)
        reader.update()

        self.__result = chigger.exodus.ExodusResult(reader, variable='temperature', viewport=[0,0,0.5,1])
        self.__result.update()

        self.__compare_x, self.__compare_y, self.__compare_T = self.create_comparison()


    def get_line(self, point_1, point_2, num_points = 200):
        # Find the temperature of a line of points
        sample = chigger.exodus.ExodusResultLineSampler(self.__result, resolution = num_points, point1 = point_1, point2 = point_2)
        sample.update()
        x = sample[0].getDistance()
        y = sample[0].getSample('temperature')
        return x, y
    

    def plot_1D(self):
        # Plot a graph of a 1D line
        x_values, y_values = self.get_line((0.013, -0.0135, 0), (0.013, 0.0215, 0))
        plt.plot(x_values, y_values)
        plt.show()
        plt.close()

    
    def get_point_temp(self, pos):
        # Get the temperature at a point
        sample = chigger.exodus.ExodusResultLineSampler(self.__result, resolution=1, point1 = pos, point2 = pos)
        sample.update()
        temp_value = sample[0].getSample('temperature')
        return torch.Tensor([temp_value[0]])


    def create_comparison(self, num_x = 40, num_y = 40):
        # Find the points on the surface of the monoblock to compare the predicted temperature & actual temperature at
        x_values = []
        y_values = []
        T_values = []
        for x in torch.linspace(X_BOUNDS[0], X_BOUNDS[1], num_x):
            for y in torch.linspace(Y_BOUNDS[0], Y_BOUNDS[1], num_y):
                if self.check_face((x, y, 0)):
                    x_values.append(x)
                    y_values.append(y)
                    T_values.append(self.get_point_temp([x, y, 0]))
        return torch.Tensor(x_values), torch.Tensor(y_values), torch.Tensor(T_values)


    def plot_3D(self):
        # Plot a smart 3D graph of the temperature at various points of the monoblock
        fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
        surf = ax.plot_trisurf(self.__compare_x, self.__compare_y, self.__compare_T, cmap=cm.jet, linewidth=0.1)
        fig.colorbar(surf, shrink=0.5, aspect=5)
        plt.show()
        plt.close()
        

    def check_face(self, pos):
        # Check if a point is not in the monoblock's hole
        if pos[0]**2 + pos[1] ** 2 < 0.006**2:
            return False
        return True

    
    def get_comparison_positions(self):
        # Return the comparison positions
        return self.__compare_x, self.__compare_y


    



if __name__ == "__main__":
    manager = ExodusManager('monoblock_out11.e')
    manager.plot_3D()
