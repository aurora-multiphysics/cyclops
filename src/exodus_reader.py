from scipy.interpolate import LinearNDInterpolator
from matplotlib import pyplot as plt
from matplotlib import cm
from tqdm import tqdm
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
    def __init__(self, file_name, face):
        # Load the file
        parent_path = os.path.dirname(os.path.dirname(__file__))
        full_path = os.path.join(os.path.sep,parent_path,'simulation', file_name)

        print('\nReading mesh...')
        mesh = meshio.read(full_path)
        mesh_positions = mesh.points
        temperatures = mesh.point_data['temperature']
        self.__interpolater = LinearNDInterpolator(mesh_positions, temperatures)

        if face == 'f':
            self.__comparison_pos, self.__comparison_temps = self.generate_front()
        else:
            self.__comparison_pos, self.__comparison_temps = self.generate_side()

        self.__mean_kernel = self.generate_kernel()
        self.__face = face


    def get_face_ID(self):
        return self.__face


    def get_positions(self):
        return self.__comparison_pos


    def get_temperatures(self):
        return self.__comparison_temps
    

    def find_temps(self, pos):
        return self.__interpolater(pos)
    

    def find_mean_temp(self, pos):
        pos_in_radius = self.__mean_kernel + np.ones(pos.shape)*pos
        print(pos_in_radius)
        temps = self.find_temps(pos_in_radius)
        return np.mean(temps)


    def generate_kernel(self, num_x = 5, num_y = 5):
        x_values = np.linspace(-THERMOCOUPLE_RADIUS, THERMOCOUPLE_RADIUS, num_x)
        y_values = np.linspace(-THERMOCOUPLE_RADIUS, THERMOCOUPLE_RADIUS, num_y)
        pos_in_radius = []
        for x in x_values:
            for y in y_values:
                if x**2 + y**2 <= THERMOCOUPLE_RADIUS**2:
                    pos = np.array([x, y])
                    pos_in_radius.append(pos)
        return np.array(pos_in_radius) 


    def generate_front(self, num_x = 20, num_y = 20):
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
        return np.concatenate((x, y), axis=1), temps
    

    def generate_side(self, num_z = 20, num_y = 20, double=True):
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
        return np.concatenate((z, y), axis=1), temps

    
    def check_front_face(self, x, y):
        # Check if a point is not in the monoblock's hole
        if x**2 + y ** 2 <= MONOBLOCK_RADIUS**2:
            return False
        return True


    def plot_3D(self, x_positions, y_positions, temp_values):
        # Plot a smart 3D graph of the temperature at various points of the monoblock
        fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
        surf = ax.plot_trisurf(x_positions, y_positions, temp_values, cmap=cm.plasma, linewidth=0.1)
        fig.colorbar(surf, shrink=0.5, aspect=5)
        plt.show()
        plt.close()
        



if __name__ == "__main__":
    exodus_reader = ExodusReader('monoblock_out.e', 's')
    positions = exodus_reader.get_positions()
    temps = exodus_reader.get_temperatures()
    exodus_reader.plot_3D(positions[:,0], positions[:,1], temps)