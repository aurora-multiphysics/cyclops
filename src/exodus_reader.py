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
RADIUS = 0.006




class ExodusReader():
    def __init__(self, file_name):
        # Load the file
        parent_path = os.path.dirname(os.path.dirname(__file__))
        full_path = os.path.join(os.path.sep,parent_path,'simulation', file_name)

        reader = chigger.exodus.ExodusReader(full_path)
        reader.update()

        # __source stores the exodus file's data
        self.__source = chigger.exodus.ExodusSource(reader)
        self.__source.update(variable='temperature')
        print(self.__source.getRange())

    
    def get_point_temp(self, pos):
        # Get the temperature from a specified
        sample = chigger.exodus.ExodusSourceLineSampler(
            self.__source, 
            resolution=1, 
            point1 = pos, 
            point2 = pos
        )
        sample.update()
        temp_value = sample.getSample('temperature')[0]
        return temp_value


    def get_grid(self, num_z = 30, num_y = 30):
        # Gets a grid of position values and their corresponding temperatures
        z_values = np.linspace(0, 0.012, num_z)
        y_values = np.linspace(Y_BOUNDS[0], Y_BOUNDS[1], num_y)

        temp = []
        x = []
        y = []

        print("\nGenerating node data...")
        for i in tqdm(range(len(z_values))):
            for j in range(len(y_values)):
                #if self.check_face(z_values[i], y_values[j]):
                rounded_z = np.round(z_values[i], 7)
                rounded_y = np.round(y_values[j], 7)
                temp.append(self.get_point_temp([0.01, rounded_y, rounded_z]))
                x.append(rounded_z)
                y.append(rounded_y)
        print(temp)
        return x, y, temp


    def send_to_csv(self, x, y, temp):
        # Stores the position and temperature data in the columns of a csv file
        data = {
            'X': x, 
            'Y': y,
            'T': temp
        }

        dataframe = pd.DataFrame(data)
        print('\n', dataframe)

        parent_path = os.path.dirname(os.path.dirname(__file__))
        full_path = os.path.join(os.path.sep,parent_path,'simulation', 'side_field.csv')
        dataframe.to_csv(full_path, index=False)


    def plot_3D(self, x_positions, y_positions, temp_values):
        # Plot a smart 3D graph of the temperature at various points of the monoblock
        fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
        surf = ax.plot_trisurf(x_positions, y_positions, temp_values, cmap=cm.jet, linewidth=0.1)
        fig.colorbar(surf, shrink=0.5, aspect=5)
        plt.show()
        plt.close()
        

    def check_face(self, x, y):
        # Check if a point is not in the monoblock's hole
        if x**2 + y ** 2 <= RADIUS**2:
            return False
        if x <= X_BOUNDS[0] or x >= X_BOUNDS[1]:
            return False
        if y <= Y_BOUNDS[0] or y >= Y_BOUNDS[1]:
            return False
        return True


    



if __name__ == "__main__":
    exodus_reader = ExodusReader('monoblock_out11.e')
    x, y, temps = exodus_reader.get_grid()
    exodus_reader.plot_3D(x, y, temps)
    exodus_reader.send_to_csv(x, y, temps)