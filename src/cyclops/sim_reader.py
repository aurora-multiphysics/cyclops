"""
MeshReader and Unfolder classes for cyclops.

Handle reading simulation data into usable planes.

(c) Copyright UKAEA 2023-2024.
"""
import alphashape
import numpy as np
import meshio
from descartes import PolygonPatch
from skspatial.objects import Plane
from scipy.spatial import Delaunay


class MeshReader:
    """Class to read mesh files using meshio."""

    def __init__(self, file_path: str) -> None:
        """Load a mesh file from the simulation folder.

        Loaded mesh is read into a private attribute __mesh.
        Method should work for a variety of mesh formats, for a full complete
        list of formats supported see: https://pypi.org/project/meshio/

        Args:
        ----
        file_path : (str) path to the mesh file e.g. 'simulation/mesh.e'.

        Returns:
        -------
        None.
        """
        self.__mesh = meshio.read(file_path)
        print("Loaded mesh: ", self.__mesh)

    def read_pos(self, set_name: str) -> np.ndarray[float]:
        """Read and record the points described by current region in a numpy
        array.

        Args:
            set_name (str): region name.

        Returns:
            np.ndarray[float]: n by d array of n positions with d dimensions.
        """
        points = []
        print("now reading positions in...")
        for point_index in self.__mesh.point_sets[set_name]:
            points.append(self.__mesh.points[point_index])
        #print("                        ")
        return np.array(points)

    def read_scalar(
        self, set_name: str, scalar_name: str
    ) -> np.ndarray[float]:
        """Find values of named scalar at the points specified by region name.

        Args:
            set_name (str): region name
            scalar_name (str): name of the scalar value to be read

        Returns:
            np.ndarray[float]: array of n scalar values
        """
        set_values = []
        all_values = self.__mesh.point_data[scalar_name]

        for point_index in self.__mesh.point_sets[set_name]:
            set_values.append(all_values[point_index])
        print("                        ")
        return np.array(set_values)


class Unfolder:
    """Class for unfolding 3D geometries with the fields/scalars measured on
    them into lower dimensional planes.

    Performs a number of operations on the position arrays produced by
    reading in a given mesh.
    """

    def BPCA_data_prep(self, loaded_mesh: MeshReader, point_sets: list(),
                       point_data: list()) -> np.ndarray[float]:
        """Prepares input data to be handled by the BPCA function, i.e.
        matrices with columns "X,Y,Z" and "I" where I is the property being
        measured, temperature for example. Iterates over the different
        sections of a given mesh and constructs a matrix of the all the points
        within it. Alongside this it will calculate the upper and lower bounds
        for each section of the mesh and create a pair of final upper and
        lower bound matrices of the same size and shape as the data matrix.

        Args:
        ----
        pos_3D : (np.ndarray[float]) n by 3 array of n 3D position vectors.

        Returns:
        -------
        reduced_matrix : (np.ndarray[float]) n by 2 array of n 2D position vectors.
        """
        
        points = []
        for pos in point_sets:
            pos_3D = loaded_mesh.read_pos(pos)
            #bounds 
        #     pos_2D.append(np.array([pos[2], pos[1]]))

        #return reduced_matrix


    def compress_2D(self, pos_3D: np.ndarray[float]) -> np.ndarray[float]:
         #To be deleted
         """Compress an array of 3D points into 2D points.

         Simple implementation by: (x, y, z) -> (z, y).

         Args:
             pos_3D (np.ndarray[float]): n by 3 array of n 3D position vectors.

         Returns:
             np.ndarray[float]: n by 2 array of n 2D position vectors.
         """
         pos_2D = []
         for pos in pos_3D:
             pos_2D.append(np.array([pos[2], pos[1]]))
         return np.array(pos_2D)

    def compress_1D(self, points: np.ndarray[float]) -> np.ndarray[float]:
        #Should be replaced by a more flexible function that calls the BPCA
         """Compress an array of 2D/3D points into 1D points.

         Works by considering the distances between each point vector.

         Args:
             points (np.ndarray[float]): n by 2 (or 3) array of n 2D (or 3D)
                 position vectors.

         Returns:
             np.ndarray[float]: n by 1 array of n 1D position vectors.
         """
         sub = np.zeros(points.shape)
         for i, pos in enumerate(points[:-1]):
             sub[i + 1] = pos
         diff = points - sub
         diff[0] = np.zeros(diff[0].shape)
         out_arr = np.sqrt((diff * diff).sum(axis=1))
         out_arr = np.cumsum(out_arr)
         return out_arr.reshape(-1, 1)

    def generate_grid(
        self, bounds: np.ndarray[float], num_x: int, num_y: int
    ) -> np.ndarray[float]:
        #Need to make 3D
        """Generate a rectangular grid of values in the bounded region.

        Args:
            bounds (np.ndarray[float]): of the form [[x1, y1], [x2, y2]].
            num_x (int): number of points in the x direction.
            num_y (int): number of points in the y direction.

        Returns:
            np.ndarray[float]: array of 2D positions.
        """
        (min_x, min_y), (max_x, max_y) = bounds
        x_values = np.linspace(min_x, max_x, num_x).reshape(-1)
        y_values = np.linspace(min_y, max_y, num_y).reshape(-1)

        grid_pos = []
        for x in x_values[1:-1]:
            for y in y_values[1:-1]:
                grid_pos.append(np.array([x, y]))
        return np.array(grid_pos)

    def generate_line(
        self, pos1: np.ndarray[float], pos2: np.ndarray[float], num_points: int
    ) -> np.ndarray[float]:
        #Not sure the purpose of this)
        """Generate a 2D line between two 3D positions.

        Args:
            pos1 (np.ndarray[float]): start position of the form [x1, y1, z1].
            pos2 (np.ndarray[float]): end position of the form [x2, y2, z2].
            num_points (int): number of points in the line.

        Returns:
            np.ndarray[float]: n by 2 array where n=num_points.
        """
        x_values = np.linspace(pos1[0], pos2[0], num_points).reshape(-1, 1)
        y_values = np.linspace(pos1[1], pos2[1], num_points).reshape(-1, 1)
        line_pos = np.concatenate((x_values, y_values), axis=1)
        return line_pos

    def find_bounds(self, point_cloud: np.ndarray) -> np.ndarray[float]:
        #Need to make 3D and cover non rectangular shapes
        """Return the rectangular bounds enclosing an array of positions.

        Args:
            pos_2D (np.ndarray): n by 2 array.

        Returns:
            np.ndarray[float]: of the form [[x1, y1], [x2, y2]].
        """
        #Find optimal polygon to encompass points
        alpha_shape = alphashape.alphashape(point_cloud)
        shape_points = alpha_shape.vertices
        
        #Get plane equation for each constituent plane
        for triangle in alpha_shape.triangles:
            plane = Plane.best_fit(triangle)
            coefficients = plane.cartesian()


        # min_x = np.min(shape_points[:, 0])
        # max_x = np.max(shape_points[:, 0])
        # min_y = np.min(shape_points[:, 1])
        # max_y = np.max(shape_points[:, 1])
        # return np.array([[min_x, min_y], [max_x, max_y]])
