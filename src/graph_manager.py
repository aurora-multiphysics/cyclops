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
        # Draw the first plot
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
        
        # Draw the second plot
        fig_2, ax_4 = plt.subplots(figsize=(8, 6))

        differences = np.abs(true_temps - model_temps)
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
    

    def draw_optimisation(self, history):
        n_evals = []
        average_loss = []
        min_loss = []

        for algo in history:
            n_evals.append(algo.evaluator.n_eval)
            opt = algo.opt

            min_loss.append(opt.get("F").min())
            average_loss.append(algo.pop.get("F").mean())

        fig, ax = plt.subplots(figsize=(8, 6))
        plt.yscale('log')
        plt.plot(n_evals, average_loss, label='average loss')
        plt.plot(n_evals, min_loss, label = 'minimum loss')
        plt.xlabel('Function evaluations')
        plt.ylabel('Function loss')
        plt.legend()
        plt.show()
        plt.close()
    

    def draw_pareto(self, numbers, results):
        # Plot the pareto front
        fig, ax = plt.subplots(figsize=(8, 6))
        plt.scatter(numbers, results, facecolors='none', edgecolors='b')
        plt.xlabel('Number of sensors')
        plt.ylabel('Loss')
        plt.title('Pareto front')
        plt.show()
        plt.close()