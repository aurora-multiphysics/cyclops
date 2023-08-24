from regressors import RBFModel, GPModel, NModel, LModel, CTModel, CSModel
from field_management import ScalarField, VectorField
from results_management import PickleManager
from matplotlib import pyplot as plt
from matplotlib import cm
import numpy as np
import meshio
import os




def compress_2D(pos_3D):
    # Takes in (x, y, z) and returns (x, y)
    pos_2D = []
    for pos in pos_3D:
        pos_2D.append(np.array([pos[2], pos[1]]))
    return np.array(pos_2D)



def generate_grid(pos_2D, num_x, num_y):
    min_x = np.min(pos_2D[:, 0])
    max_x = np.max(pos_2D[:, 0])
    min_y = np.min(pos_2D[:, 1])
    max_y = np.max(pos_2D[:, 1])

    x_values = np.linspace(min_x, max_x, num_x).reshape(-1)
    y_values = np.linspace(min_y, max_y, num_y).reshape(-1)

    grid_pos = []
    for x in x_values:
        for y in y_values:
            grid_pos.append(np.array([x, y]))
    return np.array(grid_pos)




class ExodusReader():
    def __init__(self, file_name) -> None:
        # Load the exodus file and read it to get a mesh
        print('\nReading file...')
        self.__mesh = self.__generate_mesh(file_name)
        print(self.__mesh)


    def __generate_mesh(self, file_name):
        # Convert the file to a mesh
        dir_path = os.path.dirname(os.path.dirname(__file__))
        full_path = os.path.join(os.path.sep, dir_path,'simulation', file_name)
        return meshio.read(full_path)


    def read_pos(self, set_name):
        points = []
        for point_index in self.__mesh.point_sets[set_name]:
            points.append(self.__mesh.points[point_index])
        return np.array(points)

    
    def read_scalar(self, set_name, scalar_name):
        set_values = []
        all_values = self.__mesh.point_data[scalar_name]

        for point_index in self.__mesh.point_sets[set_name]:
            set_values.append(all_values[point_index])
        return np.array(set_values)




if __name__ == '__main__':
    reader = ExodusReader('monoblock_out.e')
    pickle_manager = PickleManager()

    sensor_region = 'right'
    pos_3D = reader.read_pos(sensor_region)
    temps = reader.read_scalar(sensor_region, 'temperature')

    disp = np.array([
        reader.read_scalar(sensor_region, 'disp_x'),
        reader.read_scalar(sensor_region, 'disp_y'),
        reader.read_scalar(sensor_region, 'disp_z')
    ]).T

    pos_2D = compress_2D(pos_3D)
    min_x = np.min(pos_2D[:, 0])
    max_x = np.max(pos_2D[:, 0])
    min_y = np.min(pos_2D[:, 1])
    max_y = np.max(pos_2D[:, 1])
    bounds = ((min_x, min_y), (max_x, max_x))

    temp_field = ScalarField(LModel, bounds)
    temp_field.fit_model(pos_2D, temps)

    disp_field = VectorField(LModel, bounds)
    disp_field.fit_model(pos_2D, disp)

    pickle_manager.save_file('simulation', 'temp.obj', temp_field)
    pickle_manager.save_file('simulation', 'disp.obj', disp_field)