from matplotlib import pyplot as plt
from matplotlib import tri as tri
from matplotlib import cm
import scienceplots
import numpy as np




# This constant is for removing the middle lines across the void in the monoblock setup diagram
RADIUS = 0.006




class GraphManager():
    def __init__(self):
        plt.style.use('science')


    def draw_double_3D_temp_field(self, positions, true_temps, model_temps):
        fig, ax = plt.subplots(subplot_kw={"projection": "3d"}, figsize=(8, 6))

        ax.set_title('Model and simulation temperature fields')
        surf_1 = ax.plot_trisurf(
            positions[:,0].reshape(-1), 
            positions[:,1].reshape(-1), 
            true_temps, 
            cmap=cm.jet
        )
        surf_2 = ax.plot_trisurf(
            positions[:,0].reshape(-1), 
            positions[:,1].reshape(-1), 
            model_temps
        )
        plt.show()
        plt.close()
    

    def draw_comparison_pane(self, all_positions, sensor_positions, true_temps, model_temps):
        fig_1, (ax_1, ax_2, ax_3) = plt.subplots(1,3, figsize=(18, 7))

        ax_1.set_title('Simulation temperature field')
        cp_1 = self.plot_contour_temp_field(ax_1, all_positions, true_temps)

        ax_2.set_title('Predicted temperature field')
        cp_2 = self.plot_contour_temp_field(ax_2, all_positions, model_temps)
        self.scatter_sensor_positions(ax_2, sensor_positions)

        ax_3.set_title('Sensor layout')
        self.plot_monoblock_grid(ax_3, all_positions)
        self.scatter_sensor_positions(ax_3, sensor_positions)

        fig_1.colorbar(cp_2, ax=[ax_1, ax_2])
        
        differences = np.zeros(len(true_temps))
        for i in range(len(true_temps)):
            differences[i] = np.abs(true_temps[i] - model_temps[i])
        fig_2, ax_4 = plt.subplots(layout='constrained', figsize=(5, 6))
        cp_4 = self.plot_field_errors(ax_4, all_positions, differences)

        fig_2.colorbar(cp_4)
        
        plt.show()
        plt.close()


    def plot_contour_temp_field(self, ax, positions, temp_values):
        ax.set_xlabel('x (m)')
        ax.set_ylabel('y (m)')

        return ax.tricontourf(
            positions[:,0].reshape(-1), 
            positions[:,1].reshape(-1), 
            temp_values, 
            cmap=cm.jet, levels = 30
        )


    def scatter_sensor_positions(self, ax, sensor_positions):
        sensor_x = []
        sensor_y = []
        for i in range(len(sensor_positions)):
            sensor_x.append(sensor_positions[i, 0])
            sensor_y.append(sensor_positions[i, 1])

        ax.set_xlabel('x (m)')
        ax.set_ylabel('y (m)')
        ax.scatter(
            sensor_x, 
            sensor_y,
            s=20,
            color='black'
        )


    def plot_monoblock_grid(self, ax, positions):
        triang = tri.Triangulation(positions[:, 0], positions[:, 1])
        triang.set_mask(np.hypot(
            positions[:, 0][triang.triangles].mean(axis=1),
            positions[:, 1][triang.triangles].mean(axis=1)) 
        < RADIUS)
        ax.triplot(triang)


    def plot_field_errors(self, ax, positions, differences):
        ax.set_xlabel('x (m)')
        ax.set_ylabel('y (m)')

        return ax.tricontourf(
            positions[:,0].reshape(-1), 
            positions[:,1].reshape(-1), 
            differences, 
            cmap=cm.Blues, levels = 30
        )
    