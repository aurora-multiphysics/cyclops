from fields import ScalarField, VectorField
from file_reader import PickleManager
from regressors import LModel, CSModel
import numpy as np
import meshio
import os



class Unfolder():
    def compress_2D(self, pos_3D):
        # Takes in (x, y, z) and returns (x, y)
        pos_2D = []
        for pos in pos_3D:
            pos_2D.append(np.array([pos[2], pos[1]]))
        return np.array(pos_2D)


    def compress_1D(self, points):
        # Tales (x, y) and returns y
        diff = points - points[0]*np.ones(points.shape)
        return np.sqrt((diff*diff).sum(axis=1))


    def generate_grid(self, bounds, num_x, num_y):
        (min_x, min_y), (max_x, max_y) = bounds
        x_values = np.linspace(min_x, max_x, num_x).reshape(-1)
        y_values = np.linspace(min_y, max_y, num_y).reshape(-1)

        grid_pos = []
        for x in x_values[1:-1]:
            for y in y_values[1:-1]:
                grid_pos.append(np.array([x, y]))
        return np.array(grid_pos)

    
    def generate_line(self, pos1, pos2, num_points):
        x_values = np.linspace(pos1[0], pos2[0], num_points).reshape(-1, 1)
        y_values = np.linspace(pos1[1], pos2[1], num_points).reshape(-1, 1)
        line_pos = np.concatenate((x_values, y_values), axis=1)
        return line_pos


    def find_bounds(self, pos_2D):
        min_x = np.min(pos_2D[:, 0])
        max_x = np.max(pos_2D[:, 0])
        min_y = np.min(pos_2D[:, 1])
        max_y = np.max(pos_2D[:, 1])
        return np.array([[min_x, min_y], [max_x, max_y]])




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