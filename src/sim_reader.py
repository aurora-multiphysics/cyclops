from fields import ScalarField, VectorField
from file_reader import PickleManager
from regressors import LModel, CSModel
import numpy as np
import meshio
import os



class GridManager():
    def compress_2D(self, pos_3D):
        # Takes in (x, y, z) and returns (x, y)
        pos_2D = []
        for pos in pos_3D:
            pos_2D.append(np.array([pos[2], pos[1]]))
        return np.array(pos_2D)


    def generate_grid(self, bounds, num_x, num_y):
        (min_x, min_y), (max_x, max_y) = bounds
        x_values = np.linspace(min_x, max_x, num_x).reshape(-1)
        y_values = np.linspace(min_y, max_y, num_y).reshape(-1)

        grid_pos = []
        for x in x_values[1:-1]:
            for y in y_values[1:-1]:
                grid_pos.append(np.array([x, y]))
        return np.array(grid_pos)


    def find_bounds(self, pos_2D):
        min_x = np.min(pos_2D[:, 0])
        max_x = np.max(pos_2D[:, 0])
        min_y = np.min(pos_2D[:, 1])
        max_y = np.max(pos_2D[:, 1])
        return np.array([[min_x, min_y], [max_x, max_y]])

    
    def compress_1D(self, points):
        # Tales (x, y) and returns y
        return points[:, 1].reshape(-1, 1)


    def field_to_line(self, field, points, new_field_type):
        point_values = field.predict_values(points)
        line_points = self.compress_1D(points)
        line_bounds = np.array([[line_points[0, 0]], [line_points[-1, 0]]])
        line_field = new_field_type(CSModel, line_bounds, 1)
        line_field.fit_model(line_points, point_values)
        return line_field




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
    grid_manager = GridManager()
    
    sensor_region = 'right'
    pos_3D = reader.read_pos(sensor_region)
    temps = reader.read_scalar(sensor_region, 'temperature')

    pos_2D = grid_manager.compress_2D(pos_3D)
    bounds = grid_manager.find_bounds(pos_2D)
    grid = grid_manager.generate_grid(bounds, 30, 30)

    temp_field = ScalarField(LModel, bounds, 2)
    temp_field.fit_model(pos_2D, temps)

    line_length = 100
    line_x = np.zeros(line_length)
    line_y = np.linspace(bounds[0, 1], bounds[1, 1], line_length)
    line_points = np.concatenate((line_x.reshape(-1, 1), line_y.reshape(-1, 1)), axis=1)
    temp_line = grid_manager.field_to_line(temp_field, line_points, ScalarField)

    pickle_manager.save_file('simulation', 'field_temp_line.obj', temp_line)
    pickle_manager.save_file('simulation', 'field_temp.obj', temp_field)
    pickle_manager.save_file('simulation', 'grid.obj', grid)

    print(temp_line.get_bounds())


    # disp = np.array([
    #     reader.read_scalar(sensor_region, 'disp_x'),
    #     reader.read_scalar(sensor_region, 'disp_y'),
    #     reader.read_scalar(sensor_region, 'disp_z')
    # ]).T
    # disp_field = VectorField(LModel, bounds, 2)
    # disp_field.fit_model(pos_2D, disp)
    # pickle_manager.save_file('simulation', 'field_disp.obj', disp_field)