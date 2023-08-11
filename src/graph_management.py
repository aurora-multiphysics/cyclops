from matplotlib.backends.backend_pdf import PdfPages
from matplotlib import pyplot as plt
from matplotlib import tri as tri
from matplotlib import cm
import scienceplots
import numpy as np
import os




# This constant is for removing the middle lines across the void in the monoblock setup diagram
RADIUS = 0.006


# Function naming
# draw_ create entire window (and may also create a figure/fill in the figure)
# build_ create a figure (and may also fill in the axes)
# plot_ fill in the axes only





class GraphManager():
    def __init__(self):
        plt.style.use('science')


    def draw_double_3D_temp_field(self, positions, true_temps, model_temps):
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
        plt.show()
        plt.close()


    def create_pdf(self, all_positions, all_layouts, true_temps, model_temps, lost_sensors, face):
        manager = PDFManager('Layouts.pdf')
        for sensor_positions in all_layouts:
            fig_1 = self.build_front_compare(all_positions, sensor_positions, true_temps, model_temps, lost_sensors, face)
            manager.save_figure(fig_1)
        manager.close_file()

    
    def draw_compare(self, all_positions, sensor_positions, true_temps, model_temps, lost_sensors, face):
        fig_1 = self.build_compare(all_positions, sensor_positions, true_temps, model_temps, lost_sensors, face)
        fig_2 = self.build_sensors(all_positions, sensor_positions, true_temps, model_temps, lost_sensors, face)
        plt.show()
        plt.close()

    
    def build_compare(self, all_positions, sensor_positions, true_temps, model_temps, lost_sensors, face):
        # Draw the first plot
        fig_1, (ax_1, ax_2, ax_3) = plt.subplots(1,3, figsize=(18, 5))

        ax_1.set_title('Simulation temperature field')
        cp_1 = self.plot_contour_field(ax_1, all_positions, true_temps)
        if face == 'f':
            self.plot_circle(ax_1)

        ax_2.set_title('Predicted temperature field')
        ax_2.sharey(ax_1)
        cp_2 = self.plot_contour_field(ax_2, all_positions, model_temps)
        self.plot_sensor_positions(ax_2, sensor_positions)
        if len(lost_sensors > 1):
            self.plot_sensor_positions(ax_2, lost_sensors, pen=('white', '*'))
        if face == 'f':
            self.plot_circle(ax_2)


        ax_3.set_title('Errors in temperature field reconstruction')
        differences = np.abs(true_temps.reshape(-1) - model_temps.reshape(-1))
        cp_3 = self.plot_field_errors(ax_3, all_positions, differences)
        if face == 'f':
            self.plot_circle(ax_3)

        fig_1.colorbar(cp_2, ax=[ax_1, ax_2])
        fig_1.colorbar(cp_3)
        return fig_1


    def build_sensors(self, all_positions, sensor_positions, lost_sensors, face):
        # Draw the second plot
        fig_2, ax_4 = plt.subplots(figsize=(5, 5))

        ax_4.set_title('Sensor layout')
        if face == 'f':
            self.plot_monoblock_grid(ax_4, all_positions, True)
        else:
            self.plot_monoblock_grid(ax_4, all_positions, False)
        self.plot_sensor_positions(ax_4, sensor_positions)
        if len(lost_sensors > 1):
            self.plot_sensor_positions(ax_4, lost_sensors, pen=('white', '*'))
        return fig_2





    def plot_contour_field(self, ax, positions, field_values):
        # Plot a sincle temperature contour field
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
        ax.scatter(
            sensor_positions[:, 0], 
            sensor_positions[:, 1],
            s=200,
            color=pen[0],
            marker=pen[1]
        )

    
    def plot_circle(self, ax):
        # Produce a white circle to overlay
        circle = plt.Circle((0, 0), RADIUS, color='w')
        ax.add_patch(circle)


    def plot_monoblock_grid(self, ax, positions, block_circle):
        # Produce a grid of the monoblock potential positions
        triang = tri.Triangulation(positions[:, 0], positions[:, 1])
        if block_circle == True:
            triang.set_mask(np.hypot(
                positions[:, 0][triang.triangles].mean(axis=1),
                positions[:, 1][triang.triangles].mean(axis=1)) 
            < RADIUS)
        ax.triplot(triang)


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
    

    def draw_optimisation(self, history):
        # Draw the optimisation progress chart
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
    

    def draw_num_pareto(self, numbers, results):
        # Draw the pareto front
        fig, ax = plt.subplots(figsize=(8, 6))
        plt.scatter(numbers, results, facecolors='none', edgecolors='b')
        plt.xlabel('Number of sensors')
        plt.ylabel('Loss')
        plt.title('Pareto front')
        plt.show()
        plt.close()


    def draw_reliability_pareto(self, F):
        fig, ax = plt.subplots(figsize=(8, 6))
        plt.scatter(F[:, 0], F[:, 1], s=30, facecolors='none', edgecolors='blue')
        plt.xlabel('Reconstruction loss')
        plt.ylabel('Reconstruction reliability')
        plt.title('Pareto front')
        plt.show()
        plt.close()





class PDFManager():
    def __init__(self, name):
        parent_path = os.path.dirname(os.path.dirname(__file__))
        file_path = os.path.join(os.path.sep,parent_path, 'results', name)
        self.__pdf_file = PdfPages(file_path)


    def save_figure(self, figure):
        self.__pdf_file.savefig(figure)
    

    def close_file(self):
        self.__pdf_file.close()
    
