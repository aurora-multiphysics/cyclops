from scipy.interpolate import LinearNDInterpolator
from matplotlib import pyplot as plt
from matplotlib import cm
import numpy as np
import pickle
import meshio
import os




def compress(pos_3D):
    # Takes in (x, y, z) and returns (x, y)
    pos_2D = []
    for pos in pos_3D:
        pos_2D.append(np.array([pos[2], pos[1]]))
    return np.array(pos_2D)




def save_field(field, file_name):
    dir_path = os.path.dirname(os.path.dirname(__file__))
    full_path = os.path.join(os.path.sep, dir_path,'simulation', file_name)

    field_file = open(full_path, 'wb')
    pickle.dump(field, field_file)
    field_file.close()




class Field():
    def __init__(self) -> None:
        self._grid_pos = None
        self._grid_magnitudes = None


    def generate_grid(self, pos_2D, num_x, num_y):
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


    def plot_field(self):
        fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
        surf = ax.plot_trisurf(self._grid_pos[:,0], self._grid_pos[:,1], self._grid_magnitudes, cmap=cm.plasma, linewidth=0.1)
        fig.colorbar(surf, shrink=0.5, aspect=5)
        plt.show()
        plt.close()




class ScalarField(Field):
    def __init__(self, known_pos_2D, known_scalars, num_x=30, num_y=30) -> None:
        self.__interpolator = LinearNDInterpolator(known_pos_2D, known_scalars)
        self._grid_pos = self.generate_grid(known_pos_2D, num_x, num_y)
        self._grid_magnitudes = self.get_scalar_xy(self._grid_pos)

    
    def get_scalar_xy(self, pos_xy):
        return self.__interpolator(pos_xy)





class VectorField(Field):
    def __init__(self, known_pos_2D, known_vectors, num_x=30, num_y=30) -> None:
        self.__num_dim = len(known_vectors[0])
        self.__interpolators = self.generate_interpolators(known_pos_2D, known_vectors)
        self._grid_pos = self.generate_grid(known_pos_2D, num_x, num_y)
        self._grid_vectors = self.generate_vectors(self._grid_pos)

        self._grid_magnitudes = self.generate_magnitudes(self._grid_vectors)


    def generate_vectors(self, pos):
        vectors = []
        for p in pos:
            vectors.append(self.get_vector_xy(p))
        return np.array(vectors)


    def generate_interpolators(self, known_pos_2D, known_vectors):
        interpolators = []
        for i in range(self.__num_dim):
            interpolator = LinearNDInterpolator(known_pos_2D, known_vectors[:, i])
            interpolators.append(interpolator)
        return interpolators


    def generate_magnitudes(self, vectors):
        magnitudes = []
        for v in vectors:
            magnitudes.append(np.linalg.norm(v))
        return np.array(magnitudes)

    
    def get_vector_xy(self, pos_xy):
        vector_out = []
        for interpolator in self.__interpolators:
            vector_out.append(interpolator(pos_xy))
        return np.array(vector_out)



class ExodusReader():
    def __init__(self, file_name) -> None:
        # Load the exodus file and read it to get a mesh
        print('\nReading file...')
        self.__mesh = self.generate_mesh(file_name)
        print(self.__mesh)


    def generate_mesh(self, file_name):
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
    # reader.fit_interpolator('temperature')
    # reader.fit_interpolator('disp_x')
    # print(reader.find_scalar('temperature', (0.01, 0.01, 0)))

    pos_3D = reader.read_pos('right')
    pos_2D = compress(pos_3D)
    temps = reader.read_scalar('right', 'temperature')

    disp = np.array([
        reader.read_scalar('right', 'disp_x'),
        reader.read_scalar('right', 'disp_y'),
        reader.read_scalar('right', 'disp_z')
    ]).T

    temp_field = ScalarField(pos_2D, temps)
    disp_field = VectorField(pos_2D, disp)

    temp_field.plot_field()
    disp_field.plot_field()

    save_field(temp_field, 'temp.obj')
    save_field(disp_field, 'disp.obj')

    