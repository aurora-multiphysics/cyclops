from simulation.exodus_manager import ExodusManager
from analysis.face_model import GaussianManager
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
        gaussian_manager = self.setup_manager(sensor_positions)
        model_temps = gaussian_manager.get_T(self.__compare_positions)
        return torch.sum(torch.abs(model_temps - self.__compare_temperatures))


    def setup_manager(self, sensor_positions):
        temp_field_values = torch.zeros(len(sensor_positions), 3)
        for i, pos in enumerate(sensor_positions):
            temp_field_values[i] = (torch.cat((pos, self.__exodus_manager.get_temp(pos)), 0))
        return GaussianManager(temp_field_values)

    
    def plot_3D(self):
        # Plot a smart 3D graph of the temperature at various points of the monoblock
        fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
        surf = ax.plot_trisurf(self.__compare_positions[:,0], self.__compare_positions[:,1], self.__compare_temperatures.reshape(-1), cmap=cm.jet, linewidth=0.1)
        fig.colorbar(surf, shrink=0.5, aspect=5)
        plt.show()
        plt.close()

    
    def plot_3D_model(self, sensor_positions):
        gaussian_manager = self.setup_manager(sensor_positions)
        with torch.no_grad():
            model_temps = gaussian_manager.get_T(self.__compare_positions)
        fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
        surf1 = ax.plot_trisurf(self.__compare_positions[:,0], self.__compare_positions[:,1], self.__compare_temperatures.reshape(-1), cmap=cm.jet, linewidth=0.1)
        surf2 = ax.plot_trisurf(self.__compare_positions[:,0], self.__compare_positions[:,1], model_temps.reshape(-1))
        
        for sensor_pos in sensor_positions:
            ax.scatter(sensor_pos[0], sensor_pos[1], self.__exodus_manager.get_temp(sensor_pos), s=100, color='black')
        plt.show()
        plt.close()


    
if __name__ == "__main__":
    loss_function = LossFunction()
    #loss_function.plot_3D()
    #loss_function.plot_3D_plane(torch.Tensor([[0, 0], [0, 0.01], [0.01, 0.01]]))
    sensor_positions = torch.cat((torch.FloatTensor(10, 1).uniform_(-0.013, 0.013), torch.FloatTensor(10, 1).uniform_(-0.013, 0.025)), 1)
    loss_function.plot_3D_model(sensor_positions)
    print(loss_function.get_loss(sensor_positions))