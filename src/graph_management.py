from matplotlib.backends.backend_pdf import PdfPages
from matplotlib import pyplot as plt
from matplotlib import tri as tri
from matplotlib import cm
from constants import *
import scienceplots
import numpy as np
import warnings
import os



# Function naming
# draw_ create entire window (and may also create a figure/fill in the figure)
# build_ create a figure (and may also fill in the axes)
# plot_ fill in the axes only

warnings.filterwarnings(action='ignore', category=RuntimeWarning)



class GraphManager():
    def __init__(self):
        plt.style.use('science')


    def create_pdf(self, all_positions, all_layouts, true_temps, all_model_temps, all_lost_sensors, face, loss, chance, model_type, sensor_keys, name):
        manager = PDFManager(name)
        fig_0 = self.build_chart(chance, loss, sensor_keys)
        fig_0.suptitle('Model type: '+model_type)

        manager.save_figure(fig_0)

        sorting_matrix = []
        for i in range(len(chance)):
            sorting_matrix.append((chance[i], loss[i], all_layouts[i], all_model_temps[i], all_lost_sensors[i]))
        sorting_matrix.sort(key=lambda a: a[0], reverse=True)

        all_layouts = []
        all_model_temps = []
        all_lost_sensors = []
        loss = []
        chance = []
        for sequence in sorting_matrix:
            chance.append(sequence[0])
            loss.append(sequence[1])
            all_layouts.append(sequence[2])
            all_model_temps.append(sequence[3])
            all_lost_sensors.append(sequence[4])
        
        for i, sensor_positions in enumerate(all_layouts):
            print(sensor_keys[i])
            fig_1 = self.build_compare(all_positions, sensor_positions, true_temps, all_model_temps[i], all_lost_sensors[i], face)
            fig_1.suptitle('Layout: '+str(i)+', chance: '+str(np.round(chance[i]*100, 4))+'\%, loss: '+str(np.round(loss[i])), fontsize=16)
            manager.save_figure(fig_1)
            fig_2 = self.build_double_3D_temp_field(all_positions, true_temps, all_model_temps[i])
            manager.save_figure(fig_2)
        manager.close_file()
        plt.close()

    
    def draw_compare(self, all_positions, sensor_positions, true_temps, model_temps, lost_sensors, face):
        fig_1 = self.build_compare(all_positions, sensor_positions, true_temps, model_temps, lost_sensors, face)
        fig_2 = self.build_sensors(all_positions, sensor_positions, true_temps, model_temps, lost_sensors, face)
        plt.show()
        plt.close()

    
    def build_chart(self, chances, losses, sensor_keys):  
        fig, ax = plt.subplots(1, 1, figsize=(6, 6))
        areas = np.zeros((5,2))
        labels = [
            'No sensor failures',
            '1 sensor failure',
            '2 sensor failures',
            '3 sensor failures',
            '$>$3 sensor failures'
        ]

        for i in range(len(chances)):
            if sensor_keys[i].count('X')==0:
                if losses[i] < LOSS_LIMIT:
                    areas[0][0] += chances[i]
                else:
                    areas[0][1] += chances[i]
            elif sensor_keys[i].count('X') == 1:
                if losses[i] < LOSS_LIMIT:
                    areas[1][0] += chances[i]
                else:
                    areas[1][1] += chances[i]
            elif sensor_keys[i].count('X') == 2:
                if losses[i] < LOSS_LIMIT:
                    areas[2][0] += chances[i]
                else:
                    areas[2][1] += chances[i]
            elif sensor_keys[i].count('X') == 3:
                if losses[i] < LOSS_LIMIT:
                    areas[3][0] += chances[i]
                else:
                    areas[3][1] += chances[i]
        
        cumulative_total = sum(areas.flatten())
        areas[4][1]= 1 - cumulative_total

        cmap = plt.colormaps["tab20"]
        outer_colors = cmap([0, 2, 8, 12, 18])
        inner_colors = cmap([4, 6]*5)

        size=0.3
        wedges = ax.pie(
            areas.sum(axis=1),
            labels = labels,
            radius=1, 
            colors=outer_colors,
            wedgeprops=dict(width=size, edgecolor='w')
        )
        #print(str(round(adequate[0]*100))+' % chance of success')
        wedges_2 = ax.pie(
            areas.flatten(),
            radius=1-size, 
            colors=inner_colors,
            wedgeprops=dict(width=size, edgecolor='w')
        )
        ax.set(aspect="equal")
        return fig


    
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
            self.plot_sensor_positions(ax_2, lost_sensors, pen=('red', '*'))
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
            self.plot_sensor_positions(ax_4, lost_sensors, pen=('red', '*'))
        return fig_2


    def build_double_3D_temp_field(self, positions, true_temps, model_temps):
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
        ax.scatter(
            sensor_positions[:, 0], 
            sensor_positions[:, 1],
            s=200,
            color=pen[0],
            marker=pen[1]
        )

    
    def plot_circle(self, ax):
        # Produce a white circle to overlay
        circle = plt.Circle((0, 0), MONOBLOCK_RADIUS, color='w')
        ax.add_patch(circle)


    def plot_monoblock_grid(self, ax, positions, block_circle):
        # Produce a grid of the monoblock potential positions
        triang = tri.Triangulation(positions[:, 0], positions[:, 1])
        if block_circle == True:
            triang.set_mask(np.hypot(
                positions[:, 0][triang.triangles].mean(axis=1),
                positions[:, 1][triang.triangles].mean(axis=1)) 
            < MONOBLOCK_RADIUS)
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


    def save_reliability_pareto(self, F, name):
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.scatter(F[:, 0], F[:, 1], s=30, facecolors='none', edgecolors='blue')
        ax.set_xlabel('Reconstruction loss')
        ax.set_ylabel('Reconstruction reliability')
        ax.set_title('Pareto front')
        parent_path = os.path.dirname(os.path.dirname(__file__))
        file_path = os.path.join(os.path.sep,parent_path, 'results', name)
        fig.savefig(file_path)
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
    
