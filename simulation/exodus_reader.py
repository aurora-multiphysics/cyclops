from matplotlib import pyplot as plt
import numpy as np
import netCDF4
import os


# LAYOUT OF FILE
# Each array of coordinates as 78,400 points arrayed in a 
# non-intuitive order. The ith value of each coordinate array
# will have a temperature as described by the ith value in the
# temperature array.
# The face (for initial analysis) contains 2992 points
# We will not bother modelling the line as this would be trivial


class ExodusReader():
    def __init__(self, name):
        # Load simulation data
        relative_path = name
        absolute_path = os.path.dirname(__file__)
        full_path = os.path.join(absolute_path, relative_path)
        simulation_file = netCDF4.Dataset(full_path)

        # Get data at time t=1
        self.__x_values = np.array(simulation_file.variables['coordx'])
        self.__y_values = np.array(simulation_file.variables['coordy'])
        self.__z_values = np.array(simulation_file.variables['coordz'])
        self.__temperatures = np.array(simulation_file.variables['vals_nod_var4'][1])
        simulation_file.close()


    def calculate_scale(self, arr_T):
        # Calculate values for scaling
        max_T = np.max(arr_T)
        min_T = np.min(arr_T)
        range_T = max_T - min_T
        return min_T, range_T


    def T_to_colour(self, arr_T):
        # Convert the temperature to a corresponding shade of blue
        colour_T = []
        min_T, range_T = self.calculate_scale(arr_T)
        for T in arr_T:
            colour_T.append((0.0, 0.0, self.scale(T, min_T, range_T)))
        return colour_T
    

    def scale(self, x, min_x, range_x):
        return (x-min_x)/range_x
    

    def display_data(self, sim_f):
        print("\n")
        for key in sim_f.variables:
            print("\n")
            print(key)
            print(np.array(sim_f.variables[key]))


    def plot_3D(self):
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        colour_T = self.T_to_colour(self.__temperatures)
        ax.scatter(self.__x_values, self.__y_values, self.__z_values, s=0.1, color=colour_T)
        ax.set_xlabel('X axis')
        ax.set_ylabel('Y axis')
        ax.set_zlabel('Z axis')
        plt.show()
        plt.close()


    def front_only(self):
        # Consider the front face only (where z=0)
        face_x = []
        face_y = []
        face_T = []
        for i in range(len(self.__x_values)):
            if self.__z_values[i] == 0.0:
                face_x.append(self.__x_values[i])
                face_y.append(self.__y_values[i])
                face_T.append(self.__temperatures[i])
        return face_x, face_y, face_T

    
    def plot_2D(self, face_x, face_y, face_T):
        plt.scatter(face_x, face_y, color=self.T_to_colour(face_T), s=0.1)
        plt.show()
        plt.close()


    def line_only(self, face_x, face_y, face_T):
        # Consider the line near the edge (where x=0.0115)
        line_y = []
        line_T = []
        uncertainty = 0.0000001
        for i in range(len(face_y)):
            if face_x[i] < 0.0115 + uncertainty and face_x[i] > 0.0115 - uncertainty:
                line_y.append(face_y[i])
                line_T.append(face_T[i])
        return line_y, line_T


    def plot_1D(self, line_y, line_T):
        plt.scatter(line_y, line_T)
        plt.show()
        plt.close()




if __name__ == "__main__":
    reader = ExodusReader("monoblock_out.e")
    face_x, face_y, face_T = reader.front_only()
    reader.plot_2D(face_x, face_y, face_T)
    line_y, line_T = reader.line_only(face_x, face_y, face_T)
    reader.plot_1D(line_y, line_T)





    