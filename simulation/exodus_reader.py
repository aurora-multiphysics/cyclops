from matplotlib import pyplot as plt
import numpy as np
import netCDF4
import os




def display_data(sim_f):
    print("\n")
    for key in sim_f.variables:
        print("\n")
        print(key)
        print(np.array(sim_f.variables[key]))


def scale(x, min_x, range_x):
    return (x-min_x)/range_x


def plot_3D(x_values, y_values, z_values, colour_T):
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.scatter(x_values, y_values, z_values, s=0.1, color=colour_T)
    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    ax.set_zlabel('Z axis')
    plt.show()
    plt.close()
    

def T_to_colour(arr_T):
    colour_T = []
    for T in arr_T:
        colour_T.append((0.0, 0.0, scale(T, min_T, range_T)))
    return colour_T




if __name__ == "__main__":
    #Load simulation data
    absolute_path = os.path.dirname(__file__)
    relative_path = "monoblock_out.e"
    full_path = os.path.join(absolute_path, relative_path)
    simulation_file = netCDF4.Dataset(full_path)


    #Get data at time t=1
    x_values = np.array(simulation_file.variables['coordx'])
    y_values = np.array(simulation_file.variables['coordy'])
    z_values = np.array(simulation_file.variables['coordz'])
    temperatures = np.array(simulation_file.variables['vals_nod_var4'][1])
    simulation_file.close()


    #Calcualte values for scaling
    min_T = np.min(temperatures)
    max_T = np.max(temperatures)
    range_T = max_T - min_T
    


    #COnsider the front face only
    face_x = []
    face_y = []
    face_T = []
    for i in range(len(x_values)):
        if z_values[i] == 0.0:
            face_x.append(x_values[i])
            face_y.append(y_values[i])
            face_T.append(temperatures[i])
    
    plt.scatter(face_x, face_y, color=T_to_colour(face_T), s=0.1)
    plt.show()
    plt.close()
    