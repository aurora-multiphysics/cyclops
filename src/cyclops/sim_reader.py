"""
MeshReader and Unfolder classes for cyclops.

Handle reading simulation data into usable planes.

(c) Copyright UKAEA 2023-2024.
"""
import alphashape
import numpy as np
import meshio

# from descartes import PolygonPatch
from skspatial.objects import Plane

# from scipy.spatial import Delaunay
from random import uniform


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
        """Reads and records the points described by given region in a numpy
        array.

        Args:
        -----
        set_name : (str) the current region name.

        Returns:
        --------
        np.ndarray : (float) n by d array of n positions with d dimensions.
        """
        points = []
        print("now reading positions in...")
        for point_index in self.__mesh.point_sets[set_name]:
            points.append(self.__mesh.points[point_index])
        print("                        ")
        return np.array(points)

    def read_scalar(self, set_name: str, scalar_name: str) -> np.ndarray[float]:
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

    def find_bounds(
        self, point_cloud: np.ndarray, points_per_plane: int
    ) -> np.ndarray[float]:
        """Function finds the the upper and lower bounds enclosing an array of
        points from a given mesh.

        Args:
        -----
        point_cloud : (np.ndarray) an array of 3D points from a section of the
            overall mesh, or the full mesh if it contains only one point set.

        Returns:
        --------
        upper_bounds : (ndarray of floats) an array of the upper bound
            coordinates for the given points/mesh.
        lower_bounds : (ndarray of floats) and array of the lower bound
            coordinates for the given points/mesh.
        """
        # Find optimal polygon to encompass points
        alpha_shape = alphashape.alphashape(point_cloud)

        def triangle_side(p1, p2, a, b):
            """Function to check if a point is on the correct side of two
            lines along a triangle to be inside that triangle."""

            cp1 = np.cross(b - a, p1 - a)
            cp2 = np.cross(b - a, p2 - a)
            if np.dot(cp1, cp2) >= 0:
                return True
            else:
                return False

        def PointInTri(p, a, b, c):
            """Function to check if a point p is inside triangle abc"""
            if (
                triangle_side(p, a, b, c)
                and triangle_side(p, b, a, c)
                and triangle_side(p, c, a, b)
            ):
                return True
            else:
                return False

        # Get plane equation for each constituent plane
        for triangle in alpha_shape.triangles:
            x_vals = triangle[:, 0]
            y_vals = triangle[:, 1]
            z_vals = triangle[:, 2]
            plane = Plane.best_fit(triangle)
            # coefficients in form, (a, b, c, d) where these correspond to the
            # plane eqn ax + by + cz + d = 0
            a, b, c, d = plane.cartesian()
            plane_points = ()
            while len(plane_points) < points_per_plane:
                x = uniform(min(x_vals), max(x_vals))
                y = uniform(min(y_vals), max(y_vals))
                z = (a * x + b * y + d) / c
                tri_pt0 = (x_vals[0], y_vals[0], z_vals[0])
                tri_pt1 = (x_vals[1], y_vals[1], z_vals[1])
                tri_pt2 = (x_vals[2], y_vals[2], z_vals[2])
                new_pt = (x, y, z)
                if PointInTri(new_pt, tri_pt0, tri_pt1, tri_pt2):
                    plane_points.append(new_pt)

            z_median = np.median(point_cloud[:, 2])

        def split_bounds(plane_points: list, z_median: float):
            """Function determines which surface points will be used as upper
            and lower bounds for the mesh. Default behaviour is to split along
            the plane z = z_median, where z_median is the median value of the
            z coordinates of all the data points. User will be prompted to
            provide an alternative plane to cut along in the form of a plane
            eqn ax + by + cz + d = 0 should they wish to do so.

            Args:
            -----
            plane_points : (list of floats) a list of points on the surface of
                the mesh used to define the boundary conditions of said mesh.
            z_median : (float) the median z-coordinate value of the original
                cloud of data points.

            Returns:
            --------
            upper_bounds : (ndarray of floats) an array of the upper bound
                coordinates for the given mesh.
            lower_bounds : (ndarray of floats) and array of the lower bound
                coordinates for the given mesh.
            """

            # Determining plane eqn to cut along
            use_nondefault = input(
                "Would you like to chose a plane to split \
                                   the boundary conditions on? Enter 'y' \
                                   for yes or 'n' for no."
            )
            if use_nondefault == "y":
                a, b, c, d = input(
                    "Please enter the coefficients a, b, c, d \
                                   for the plane equation, ax + by + cz + d =\
                                   0 With each value separated by a comma"
                )
            elif use_nondefault == "n":
                a, b, c, d = 0, 0, z_median, 0

            upper_bounds = []
            lower_bounds = []

            for entry in plane_points:
                if entry[0][0] >= a and entry[0][1] >= b and entry[0][2] >= c:
                    upper_bounds.append(entry)
                else:
                    lower_bounds.append(entry)

            return upper_bounds, lower_bounds

        upper_boundaries, lower_boundaries = split_bounds(plane_points,
                                                          z_median)
        return (upper_boundaries, lower_boundaries)


class Dim_reduction:
    """Class for reducing the dimensionality of 3D geometries into lower
    dimensional planes.

    Operates on the data points of a loaded meshio mesh and the associated
    boundaries in order to reduce the number of dimensions.
    """

    def BPCA_data_prep(
        self,
        loaded_mesh: MeshReader,
        point_sets: list(),
        measureable: str,
        vec_or_scale: str,
        k: float,
    ) -> np.ndarray[float]:
        """Prepares input data to be handled by the BPCA function, i.e.
        matrices with columns "X,Y,Z" and "I" where I is the property being
        measured, temperature for example. Iterates over the different
        sections of a given mesh and constructs a matrix of the all the points
        within it. Alongside this it will calculate the upper and lower bounds
        for each section of the mesh and create a pair of final upper and
        lower bound matrices of the same size and shape as the data matrix.

        Args:
        -----
        loaded_mesh : (meshio file that has been read in)
            sim_reader.MeshReader object containing 3D points and associated
            values.
        point_sets : (list of strings) the names of the different sections of
            the data mesh, needed to access said data.
        measurable : (string) the name of the measured property, i.e.
            'temperature'
        vec_or_scale : (string) either 'vector' or 'scalar', indicating the
            type of field the 'measureable' is.
        k : (float) the number of points to generate on each triangle making
            up the shape of the mesh. These points will serve as boundary
            conditions for those triangles. A larger k will be more
            representative but also take longer to calculate.

        Returns:
        --------
        reduced_matrix : (np.ndarray[float]) n by 2 array of n 2D position vectors.
        """
        # Need to implement 'read_vector' correctly in MeshReader so that
        # vectors can be processed.
        data_matrix = np.c_()
        upper_bound_matrix = np.c_()
        lower_bound_matrix = np.c_()
        measured_field = np.c_()
        for pos in point_sets:
            pos_3D = loaded_mesh.read_pos(pos)
            if vec_or_scale == "scalar":
                field = MeshReader.read_scalar(pos, measureable).reshape(-1, 1)
            measured_field.append(field)
            # elif vec_or_scale == 'vector':
            #     field = MeshReader.read_vector(pos, measureable)
            bounds = Unfolder.find_bounds(pos_3D, k)

            upper_bound_matrix.append(bounds[0])
            lower_bound_matrix.append(bounds[1])
            data_matrix.append(pos_3D)

        data_matrix = np.array(data_matrix)
        upper_bound_matrix = np.array(upper_bound_matrix)
        lower_bound_matrix = np.array(lower_bound_matrix)
        measured_field = np.array(measured_field)

        # return reduced_matrix
