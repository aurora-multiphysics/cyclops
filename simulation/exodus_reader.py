from matplotlib import pyplot as plt
from matplotlib import cm
from tqdm import tqdm
import pandas as pd
import numpy as np
import chigger
import os



#Monoblock values
X_BOUNDS = (-0.0135, 0.0135)
Y_BOUNDS = (-0.0135, 0.0215)
Z_BOUNDS = (0, 0.012)



class ExodusReader():
    def __init__(self, relative_path):
        # Load the file
        absolute_path = os.path.dirname(__file__)
        full_path = os.path.join(absolute_path, relative_path)
        reader = chigger.exodus.ExodusReader(full_path)
        reader.update()

        self.__source = chigger.exodus.ExodusSource(reader, variable='temperature', viewport=[0,0,0.5,1])
        self.__source.update()


    def get_line(self, point_1, point_2, num_points = 200):
        # Find the temperature of a line of points
        sample = chigger.exodus.ExodusSourceLineSampler(self.__source, resolution = num_points, point1 = point_1, point2 = point_2)
        sample.update()
        x = sample.getDistance()
        y = sample.getSample('temperature')
        return x, y
    

    def plot_1D(self):
        # Plot a graph of a 1D line
        x_values, y_values = self.get_line((0.013, -0.0135, 0), (0.013, 0.0215, 0))
        plt.plot(x_values, y_values)
        plt.show()
        plt.close()

    
    def get_point_temp(self, pos):
        # Get the temperature from a point that's a tensor
        sample = chigger.exodus.ExodusSourceLineSampler(self.__source, resolution=1, point1 = pos, point2 = pos)
        sample.update()
        temp_value = sample.getSample('temperature')[0]
        return temp_value


    def send_to_csv(self, num_x = 400, num_y = 400):
        step_x = (X_BOUNDS[1] - X_BOUNDS[0])/num_x
        step_y = (Y_BOUNDS[1] - Y_BOUNDS[0])/num_y

        x1_values = np.linspace(X_BOUNDS[0], X_BOUNDS[1] - step_x, num_x)
        x2_values = np.linspace(X_BOUNDS[0] + step_x, X_BOUNDS[1], num_x)
        y1_values = np.linspace(Y_BOUNDS[0], Y_BOUNDS[1] - step_y, num_y)
        y2_values = np.linspace(Y_BOUNDS[0] + step_y, Y_BOUNDS[1], num_y)

        temp_1 = []
        temp_3 = []
        temp_4 = []

        x1 = []
        y1 = []
        x2 = []
        y2 = []

        print("\nGenerating csv data...")
        for i in tqdm(range(len(x1_values))):
            for j in range(len(y1_values)):
                temp_1.append(self.get_point_temp([x1_values[i], y1_values[j], 0]))
                temp_3.append(self.get_point_temp([x2_values[i], y1_values[j], 0]))
                temp_4.append(self.get_point_temp([x1_values[i], y2_values[j], 0]))

                x1.append(x1_values[i])
                y1.append(y1_values[j])
                x2.append(x2_values[i])
                y2.append(y2_values[j])
        
        data = {
            'X1': x1, 
            'Y1': y1,
            'X2': x2,
            'Y2': y2,
            'T1': temp_1,
            'T3': temp_3,
            'T4': temp_4
        }

        dataframe = pd.DataFrame(data)
        print('\n', dataframe)

        absolute_path = os.path.dirname(__file__)
        full_path = os.path.join(absolute_path, 'temperature_field.csv')
        dataframe.to_csv(full_path, index=False)


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
    reader = ExodusReader('monoblock_out11.e')
    print(reader.get_point_temp([0.01, 0.015, 0.0]))
    reader.send_to_csv()
