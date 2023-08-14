from scipy.interpolate import LinearNDInterpolator
from matplotlib import pyplot as plt
from matplotlib import cm
from tqdm import tqdm
import pandas as pd
import numpy as np
import meshio
import os


#Monoblock values
X_BOUNDS = (-0.0135, 0.0135)
Y_BOUNDS = (-0.0135, 0.0215)
Z_BOUNDS = (0, 0.012)
MONOBLOCK_RADIUS = 0.006
THERMOCOUPLE_RADIUS = 0.000075

# Function naming
# get_ means getter
# find_ means calculate that value
# generate_ is for producing objects in the initialisation
# build_ is for producing objects that are used all over the place




class ExodusReader():
    def __init__(self, file_name) -> None:
        # Load the exodus file and read it
        parent_path = os.path.dirname(os.path.dirname(__file__))
        full_path = os.path.join(os.path.sep,parent_path,'simulation', file_name)

        print('\nReading mesh...')
        mesh = meshio.read(full_path)
        mesh_positions = mesh.points
        temperatures = mesh.point_data['temperature']
        self.__interpolater = LinearNDInterpolator(mesh_positions, temperatures)
    

    def find_temps(self, pos) -> np.ndarray:
        return self.__interpolater(pos)


    def generate_front(self, num_x = 20, num_y = 20) -> tuple:
        # Returns the grid of position and temperature values at the front of the monoblock
        x_values = np.linspace(X_BOUNDS[0], X_BOUNDS[1], num_x)
        y_values = np.linspace(Y_BOUNDS[0], Y_BOUNDS[1], num_y)

        temps = []
        x = []
        y = []

        print("Generating node data...")
        for i in tqdm(range(len(x_values))):
            for j in range(len(y_values)):
                if self.check_front_face(x_values[i], y_values[j]):
                    rounded_x = np.round(x_values[i], 7)
                    rounded_y = np.round(y_values[j], 7)
                    temps.append(self.find_temps(np.array([rounded_x, rounded_y, 0]))[0])
                    x.append(rounded_x)
                    y.append(rounded_y)
        
        x = np.array(x).reshape(-1, 1)
        y = np.array(y).reshape(-1, 1)
        return (np.concatenate((x, y), axis=1), temps)
    

    def generate_side(self, num_z = 20, num_y = 20, double=True) -> tuple:
        # Returns the grid of position and temperature values at the front of the monoblock
        z_values = np.linspace(Z_BOUNDS[0], Z_BOUNDS[1], num_z)
        y_values = np.linspace(Y_BOUNDS[0], Y_BOUNDS[1], num_y)

        temps = []
        z = []
        y = []

        print("Generating node data...")
        for i in tqdm(range(len(z_values))):
            for j in range(len(y_values)):
                rounded_z = np.round(z_values[i], 7)
                rounded_y = np.round(y_values[j], 7)
                temp = self.find_temps(np.array([0, rounded_y, rounded_z]))[0]
                temps.append(temp)
                z.append(rounded_z)
                y.append(rounded_y)
                if double == True and rounded_z != 0.0:
                    z.append(-rounded_z)
                    y.append(rounded_y)
                    temps.append(temp)

        z = np.array(z).reshape(-1, 1)
        y = np.array(y).reshape(-1, 1)
        return (np.concatenate((z, y), axis=1), temps)

    
    def check_front_face(self, x, y) -> bool:
        # Check if a point is not in the monoblock's hole
        if x**2 + y ** 2 <= MONOBLOCK_RADIUS**2:
            return False
        return True


    def send_to_csv(self, x, y, temp, csv_name) -> None:
        # Stores the position and temperature data in the columns of a csv file
        data = {
            'X': x, 
            'Y': y,
            'T': temp
        }

        dataframe = pd.DataFrame(data)
        print('\n', dataframe)

        parent_path = os.path.dirname(os.path.dirname(__file__))
        full_path = os.path.join(os.path.sep,parent_path,'simulation', csv_name)
        dataframe.to_csv(full_path, index=False)


    def plot_3D(self, x_positions, y_positions, temp_values) -> None:
        # Plot a smart 3D graph of the temperature at various points of the monoblock
        fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
        surf = ax.plot_trisurf(x_positions, y_positions, temp_values, cmap=cm.plasma, linewidth=0.1)
        fig.colorbar(surf, shrink=0.5, aspect=5)
        plt.show()
        plt.close()



if __name__ == '__main__':
    exodus_reader = ExodusReader('monoblock_out.e')
    pos, temps = exodus_reader.generate_side()
    exodus_reader.plot_3D(pos[:,0], pos[:,1], temps)
    exodus_reader.send_to_csv(pos[:,0], pos[:,1], temps, 'side_field.csv')