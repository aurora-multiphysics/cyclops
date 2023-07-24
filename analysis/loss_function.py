from simulation.exodus_manager import ExodusManager
from analysis.face_model import PlaneModel
from matplotlib import pyplot as plt
from matplotlib import cm
import torch




#Monoblock values
X_BOUNDS = (-0.013, 0.013)
Y_BOUNDS = (-0.013, 0.025)
Z_BOUNDS = (0, 0.012)
RADIUS = 0.00601


class LossFunction():
    def __init__(self):
        self.__exodus_manager = ExodusManager('temperature_field.csv')
        self.__compare_positions = self.get_comparison_positions()
        self.__compare_temperatures = self.get_comparison_temperatures()
        


    def get_comparison_positions(self, num_x=40, num_y=40):
        positions = []
        for x in torch.linspace(X_BOUNDS[0], X_BOUNDS[1], num_x):
            for y in torch.linspace(Y_BOUNDS[0], Y_BOUNDS[1], num_y):
                if self.check_pos([x, y]):
                    positions.append([x, y])
        return torch.Tensor(positions)


    def check_pos(self, pos):
        if pos[0] <= X_BOUNDS[0] or pos[0] >= X_BOUNDS[1]:
            return False
        if pos[1] <= Y_BOUNDS[0] or pos[1] >= Y_BOUNDS[1]:
            return False
        if (pos[0]**2) + (pos[1]**2) <= RADIUS ** 2:
            return False
        return True


    def get_comparison_temperatures(self):
        temps = []
        for pos in self.__compare_positions:
            temps.append(self.__exodus_manager.get_temp(pos))
        return torch.Tensor(temps)

    
    def get_loss(self, sensor_positions):
        plane = self.setup_plane(sensor_positions)
        plane_temps = plane.get_T(self.__compare_positions)
        return torch.sum(torch.abs(plane_temps - self.__compare_temperatures))


    def setup_plane(self, sensor_positions):
        pos_1 = torch.cat((sensor_positions[0], self.__exodus_manager.get_temp(sensor_positions[0])))
        pos_2 = torch.cat((sensor_positions[1], self.__exodus_manager.get_temp(sensor_positions[1])))
        pos_3 = torch.cat((sensor_positions[2], self.__exodus_manager.get_temp(sensor_positions[2])))
        return PlaneModel(pos_1, pos_2, pos_3)

    
    def plot_3D(self):
        # Plot a smart 3D graph of the temperature at various points of the monoblock
        fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
        surf = ax.plot_trisurf(self.__compare_positions[:,0], self.__compare_positions[:,1], self.__compare_temperatures.reshape(-1), cmap=cm.jet, linewidth=0.1)
        fig.colorbar(surf, shrink=0.5, aspect=5)
        plt.show()
        plt.close()

    
    def plot_3D_plane(self, sensor_positions):
        plane = self.setup_plane(sensor_positions)
        plane_temps = plane.get_T(self.__compare_positions)

        fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
        surf1 = ax.plot_trisurf(self.__compare_positions[:,0], self.__compare_positions[:,1], self.__compare_temperatures.reshape(-1), cmap=cm.jet, linewidth=0.1)
        surf2 = ax.plot_trisurf(self.__compare_positions[:,0], self.__compare_positions[:,1], plane_temps.reshape(-1))
        ax.scatter(sensor_positions[0, 0], sensor_positions[0, 1], self.__exodus_manager.get_temp(sensor_positions[0]), s=100, color='black', edgecolor='black')
        ax.scatter(sensor_positions[1, 0], sensor_positions[1, 1], self.__exodus_manager.get_temp(sensor_positions[1]), s=100, color='black', edgecolor='black')
        ax.scatter(sensor_positions[2, 0], sensor_positions[2, 1], self.__exodus_manager.get_temp(sensor_positions[2]), s=100, color='black', edgecolor='black')
        
        plt.show()
        plt.close()


    
if __name__ == "__main__":
    loss_function = LossFunction()
    #loss_function.plot_3D()
    loss_function.plot_3D_plane(torch.Tensor([[0, 0], [0, 0.01], [0.01, 0.01]]))