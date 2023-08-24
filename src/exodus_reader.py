from simulation_management import ScalarField, VectorField, ScalarLine, VectorLine
from results_management import PickleManager
import numpy as np
import pickle
import meshio
import os




def compress_2D(pos_3D):
    # Takes in (x, y, z) and returns (x, y)
    pos_2D = []
    for pos in pos_3D:
        pos_2D.append(np.array([pos[2], pos[1]]))
    return np.array(pos_2D)




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
    temp_field = ScalarField(pos_2D, temps)
    disp_field = VectorField(pos_2D, disp)

    temp_field.draw()
    disp_field.draw()

    pickle_manager.write_file('simulation', 'temp.obj', temp_field)
    pickle_manager.write_file('simulation', 'disp.obj', disp_field)

    x = np.linspace(1, 100, 10).reshape(-1, 1)
    test_line = ScalarLine(x, np.sqrt(x))
    test_line.draw()