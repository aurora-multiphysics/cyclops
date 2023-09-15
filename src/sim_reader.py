import numpy as np
import meshio
import os


class Unfolder:
    """
    Performs a number of operations on the arrays of positions produced by
        reading the mesh.
    Will later be generalised to unfold 3D meshes into 2D planes, and produce
        an array of boundaries describing their boundaries.
    """

    def compress_2D(self, pos_3D: np.ndarray[float]) -> np.ndarray[float]:
        """
        Compresses an array of 3D points into 2D points.
        Currently it does this by going (x, y, z) -> (z, y).

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
        """
        Compresses an array of 2D/3D points into 1D points.
        Does this by considering the distances between each point vector.

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
        """
        Generates a grid of values in the rectangular region described by the
            bounds.

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
        """
        Generates a 2D line between 2 3D positions.

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

    def find_bounds(self, pos_2D: np.ndarray) -> np.ndarray[float]:
        """
        Returns the rectangular bounds enclosing an array of positions.

        Args:
            pos_2D (np.ndarray): n by 2 array.

        Returns:
            np.ndarray[float]: of the form [[x1, y1], [x2, y2]].
        """
        min_x = np.min(pos_2D[:, 0])
        max_x = np.max(pos_2D[:, 0])
        min_y = np.min(pos_2D[:, 1])
        max_y = np.max(pos_2D[:, 1])
        return np.array([[min_x, min_y], [max_x, max_y]])


class MeshReader:
    def __init__(self, file_name: str) -> None:
        """
        Load a mesh file from the simulation folder and read it into a private
            attribute __mesh.
        It will work for a variety of mesh formats, see the link below for the
            full list.
        https://pypi.org/project/meshio/

        Args:
            file_name (str): the name of the mesh file, eg 'mesh.e'.
        """
        self.__mesh = self.__generate_mesh(file_name)
        print(self.__mesh)

    def __generate_mesh(self, file_name: str) -> meshio.Mesh:
        """
        Reads in the file name, generates the absolute path to the file.
        Then returns the mesh.

        Args:
            file_name (str): name of the mesh.

        Returns:
            meshio.Mesh: mesh object.
        """
        dir_path = os.path.dirname(os.path.dirname(__file__))
        full_path = os.path.join(
            os.path.sep, dir_path, "simulation", file_name
        )
        return meshio.read(full_path)

    def read_pos(self, set_name: str) -> np.ndarray[float]:
        """
        Records the points described by the region specified into a numpy
            array.

        Args:
            set_name (str): region name.

        Returns:
            np.ndarray[float]: n by d array of n positions with d dimensions.
        """
        points = []
        for point_index in self.__mesh.point_sets[set_name]:
            points.append(self.__mesh.points[point_index])
        return np.array(points)

    def read_scalar(
        self, set_name: str, scalar_name: str
    ) -> np.ndarray[float]:
        """
        Finds the values of the scalar named at the points specified by region
            name.

        Args:
            set_name (str): region name
            scalar_name (str): name of the scalar value we are finding

        Returns:
            np.ndarray[float]: n long array of n scalar values
        """
        set_values = []
        all_values = self.__mesh.point_data[scalar_name]

        for point_index in self.__mesh.point_sets[set_name]:
            set_values.append(all_values[point_index])
        return np.array(set_values)
