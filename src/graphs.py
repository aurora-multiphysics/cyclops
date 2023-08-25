from matplotlib import pyplot as plt
from matplotlib import tri as tri
from matplotlib import cm
import scienceplots
import numpy as np


class GraphManager():
    def __init__(self):
        plt.style.use('science')
    

    def draw(self, figure):
        plt.show()
        plt.close()

    
    def build_1D_compare(self, all_positions, sensor_positions, sensor_values, true_field, model_field):
        # Draw the first plot
        fig_1, (ax_1, ax_2) = plt.subplots(1,2, figsize=(18, 5))

        ax_1.set_title('Temperature fields')
        # Plot lines of both temperature fields
        self.plot_line_fields(ax_1, all_positions, sensor_positions, sensor_values, true_field, model_field)
        ax_1.set_xlabel('x (m)')
        ax_1.set_ylabel('T (C)')


        ax_2.set_title('Errors in temperature field reconstruction')
        differences = np.abs(model_field - true_field)
        ax_2.plot(all_positions.reshape(-1), differences.reshape(-1))
        ax_2.set_xlabel('x (m)')
        ax_2.set_ylabel('difference (C)')


    def build_2D_compare(self, all_positions, sensor_positions, true_temps, model_temps):
        # Draw the first plot
        fig_1, (ax_1, ax_2, ax_3) = plt.subplots(1,3, figsize=(18, 5))

        ax_1.set_title('Simulation temperature field')
        cp_1 = self.plot_contour_field(ax_1, all_positions, true_temps)

        ax_2.set_title('Predicted temperature field')
        ax_2.sharey(ax_1)
        cp_2 = self.plot_contour_field(ax_2, all_positions, model_temps)
        self.plot_sensor_positions(ax_2, sensor_positions)

        ax_3.set_title('Errors in temperature field reconstruction')
        differences = np.abs(true_temps - model_temps)
        cp_3 = self.plot_field_errors(ax_3, all_positions, differences.reshape(-1))

        fig_1.colorbar(cp_2, ax=[ax_1, ax_2])
        fig_1.colorbar(cp_3)
        return fig_1


    def build_3D_compare(self, positions, true_temps, model_temps):
        # Draw the 3D double temperature field
        fig, ax = plt.subplots(subplot_kw={"projection": "3d"}, figsize=(8, 6))

        ax.set_title('Model and simulation temperature fields')
        surf_1 = ax.plot_trisurf(
            positions[:,0].reshape(-1), 
            positions[:,1].reshape(-1), 
            true_temps.reshape(-1), 
            cmap=cm.plasma
        )
        surf_2 = ax.plot_trisurf(
            positions[:,0].reshape(-1), 
            positions[:,1].reshape(-1), 
            model_temps.reshape(-1)
        )
        return fig

    
    def plot_line_fields(self, ax, positions, sensor_positions, sensor_values, true_field, model_field, pen=('black', '*')):
        ax.plot(positions, true_field, label='True field')
        ax.plot(positions, model_field, label = 'Predicted field')
        ax.scatter(
            sensor_positions, 
            sensor_values, 
            s=20,
            color=pen[0],
            marker=pen[1]
        )
        ax.legend()


    def plot_contour_field(self, ax, positions, field_values):
        # Plot a single temperature contour field
        ax.set_xlabel('x (m)')
        ax.set_ylabel('y (m)')

        return ax.tricontourf(
            positions[:,0].reshape(-1), 
            positions[:,1].reshape(-1), 
            field_values.reshape(-1), 
            cmap=cm.plasma, 
            levels=np.linspace(100, 1600, 30)
        )


    def plot_sensor_positions(self, ax, sensor_positions, pen=('black', 'o')):
        # Produce a scatter plot of the sensor positions to overlay
        return ax.scatter(
            sensor_positions[:, 0], 
            sensor_positions[:, 1],
            s=200,
            color=pen[0],
            marker=pen[1]
        )


    def plot_field_errors(self, ax, positions, differences):
        # Plot the difference in field temperatures
        ax.set_xlabel('x (m)')
        ax.set_ylabel('y (m)')

        return ax.tricontourf(
            positions[:,0], 
            positions[:,1], 
            differences, 
            cmap=cm.Blues, levels = 30
        )
    

    def get_optimisation_history(self, history):
        n_evals = []
        average_loss = []
        min_loss = []

        for algo in history:
            n_evals.append(algo.evaluator.n_eval)
            opt = algo.opt
            min_loss.append(opt.get("F")[:,0].min())
            average_loss.append(algo.pop.get("F")[:,0].mean())
        return n_evals, average_loss, min_loss


    def build_optimisation(self, history):
        # Draw the optimisation progress chart
        n_evals, average_loss, min_loss = self.get_optimisation_history(history)
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.set_yscale('log')
        ax.plot(n_evals, average_loss, label='average loss')
        ax.plot(n_evals, min_loss, label = 'minimum loss')
        ax.set_xlabel('Function evaluations')
        ax.set_ylabel('Loss')
        ax.set_title('Optimisation')
        ax.legend()


    def build_pareto(self, F):
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.scatter(F[:, 0], F[:, 1], s=30, facecolors='none', edgecolors='blue')
        ax.set_xlabel('Reconstruction loss')
        ax.set_ylabel('Reconstruction reliability')
        ax.set_title('Pareto front')