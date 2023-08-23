from scipy.interpolate import LinearNDInterpolator, CubicSpline
from matplotlib import pyplot as plt
from matplotlib import cm
import numpy as np





class Field():
    def __init__(self) -> None:
        self._grid_pos = None
        self._grid_magnitudes = None


    def _generate_grid(self, pos_2D, num_x, num_y):
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


    def draw(self):
        fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
        surf = ax.plot_trisurf(self._grid_pos[:,0], self._grid_pos[:,1], self._grid_magnitudes, cmap=cm.plasma, linewidth=0.1)
        fig.colorbar(surf, shrink=0.5, aspect=5)
        plt.show()
        plt.close()
    
    
    def get_bounds(self):
        min_x = np.min(self._grid_pos[:,0])
        max_x = np.max(self._grid_pos[:,0])
        min_y = np.min(self._grid_pos[:,1])
        max_y = np.max(self._grid_pos[:,1])
        return ((min_x, min_y), (max_x, max_x))




class Line():
    def __init__(self) -> None:
        self._line_pos = None
        self._line_magnitudes = None


    def _generate_strip(self, pos_1D, num_x):
        min_x = np.min(pos_1D)
        max_x = np.max(pos_1D)
        return np.linspace(min_x, max_x, num_x)


    def draw(self):
        plt.plot(self._line_pos, self._line_magnitudes)
        plt.show()
        plt.close()

    
    def get_bounds(self):
        min_x = np.min(self._line_pos)
        max_x = np.max(self._line_pos)
        return (min_x, max_x)



class ScalarInterpolator():
    def __init__(self) -> None:
        self._interpolator = None

    
    def get_scalar(self, pos):
        return self._interpolator(pos)



class VectorInterpolator():
    def __init__(self) -> None:
        self._interpolators = None
        

    def _generate_interpolators(self, known_pos, known_vectors, interp_type):
        interpolators = []
        for i in range(self._num_dim):
            interpolator = interp_type(known_pos, known_vectors[:, i])
            interpolators.append(interpolator)
        return interpolators


    def _generate_magnitudes(self, vectors):
        magnitudes = []
        for v in vectors:
            magnitudes.append(np.linalg.norm(v))
        return np.array(magnitudes)

    
    def get_vector(self, pos):
        vector_out = []
        for i, interpolator in enumerate(self._interpolators):
            vector_out.append(interpolator(pos))
        return np.array(vector_out).T




class ScalarLine(Line, ScalarInterpolator):
    def __init__(self, known_pos_1D, known_scalars, num_x=30) -> None:
        Line.__init__(self)
        ScalarInterpolator.__init__(self)
        self._interpolator = CubicSpline(known_pos_1D.reshape(-1), known_scalars.reshape(-1))

        self._line_pos = self._generate_strip(known_pos_1D, num_x)
        self._line_magnitudes = self.get_scalar(self._line_pos)




class VectorLine(Line, VectorInterpolator):
    def __init__(self, known_pos_1D, known_vectors, num_x=30) -> None:
        self._num_dim = len(known_vectors[0])
        Line.__init__(self)
        VectorInterpolator.__init__(self)
        self._interpolators = self._generate_interpolators(known_pos_1D, known_vectors, CubicSpline)
        
        self._line_pos = self._generate_strip(known_pos_1D, num_x)
        self._line_vectors = self.get_vector(self._line_pos)
        self._line_magnitudes = self._generate_magnitudes(self._line_vectors)




class ScalarField(Field, ScalarInterpolator):
    def __init__(self, known_pos_2D, known_scalars, num_x=30, num_y=30) -> None:
        Field.__init__(self)
        ScalarInterpolator.__init__(self)
        self._interpolator = LinearNDInterpolator(known_pos_2D, known_scalars)

        self._grid_pos = self._generate_grid(known_pos_2D, num_x, num_y)
        self._grid_magnitudes = self.get_scalar(self._grid_pos)




class VectorField(Field, VectorInterpolator):
    def __init__(self, known_pos_2D, known_vectors, num_x=30, num_y=30) -> None:
        self._num_dim = len(known_vectors[0])
        Field.__init__(self)
        VectorInterpolator.__init__(self)
        self._interpolators = self._generate_interpolators(known_pos_2D, known_vectors, LinearNDInterpolator)


        self._grid_pos = self._generate_grid(known_pos_2D, num_x, num_y)
        self._grid_vectors = self.get_vector(self._grid_pos)
        self._grid_magnitudes = self._generate_magnitudes(self._grid_vectors)