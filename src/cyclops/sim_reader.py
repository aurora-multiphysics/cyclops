"""
MeshReader and Unfolder classes for cyclops.

Handle reading simulation data into usable planes.

(c) Copyright UKAEA 2023.
"""
import numpy as np
import meshio
import pyvista as pv

from collections import Counter
from random import shuffle

class MeshReader:
    """Class to read mesh files using meshio."""

    def __init__(self, file_path: str) -> None:
        """Load a mesh file from the simulation folder.

        Loaded mesh is read into a private attribute __mesh.
        It will work for a variety of mesh formats, for the full list see:
        https://pypi.org/project/meshio/

        Args:
            file_path (str): path to the mesh file e.g. 'simulation/mesh.e'.
        """
        self.__mesh = meshio.read(file_path)

    def read_pos(self, set_name: str) -> np.ndarray[float]:
        """Record the points described by the region into a numpy array. Mesh
        may be split into 'sets', it is best to read points in by set so that 
        the scalar/vector values for each set are easily matched up.

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
        """Find values of named scalar at the points specified by region name.

        Args:
            set_name (str): region name
            scalar_name (str): name of the scalar value to read

        Returns:
            np.ndarray[float]: n long numpy array of n scalar values
        """
        set_values = []
        all_values = self.__mesh.point_data[scalar_name]

        for point_index in self.__mesh.point_sets[set_name]:
            set_values.append(all_values[point_index])
        
        #print(list(self.__mesh.keys()))

        return np.array(set_values)
    
    def generate_grid(self, resolution: int) -> np.ndarray[float]:
        """Generate a grid of values in the region bounded by a given mesh.

        Args:
            resolution (int): the resolution to generate the grid at, the
            higher the value the finer the grid. (Note that this is not scaled to
            the mesh, the resolution should be adjusted for the size of the mesh in
            questioned)

        Returns:
            np.ndarray[float]: array of grid point positions.
        """
        faces = self.__mesh.cells_dict
        vertices = MeshReader.read_points(self)

        # Create a PyVista mesh from the vertices and faces
        pyvista_mesh = pv.PolyData(vertices, faces)

        # Get the bounding box of the mesh
        min_bound = vertices.min(axis=0)
        max_bound = vertices.max(axis=0)

        # Create the 3D grid points using numpy's linspace
        x = np.linspace(min_bound[0], max_bound[0], resolution)
        y = np.linspace(min_bound[1], max_bound[1], resolution)
        z = np.linspace(min_bound[2], max_bound[2], resolution)

        # Generate grid
        grid_x, grid_y, grid_z = np.meshgrid(x, y, z)
        grid_points = np.vstack([grid_x.ravel(), grid_y.ravel(),
                                 grid_z.ravel()]).T

        # To do - check if there is a better method for this
        def is_point_in_mesh(point, mesh):
            # Using pyvista's method
            return mesh.is_point_in_mesh(point)

        # Filter grid points that lie inside the mesh
        valid_points = [point for point in grid_points if is_point_in_mesh(point, pyvista_mesh)]

        grid_pos = np.array(valid_points)

        return grid_pos

    def generate_line(
        self, pos1: np.ndarray[float], pos2: np.ndarray[float], num_points: int
    ) -> np.ndarray[float]:
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
    
    def get_element_faces(element, element_type):
        """ Return the faces of an element, given that element's type, where
        the element is a cell from a mesh. 
        
        Args: 
            element (mesh cell): an individual cell which forms part of a
            meshio compatible mesh.
            element_type : the type/shape of the cell provided.
        
        Returns:
            faces : a list of the faces that make up the element originally
            provided to the function, (in the form of a list of the points
            which define that face).
            """
        faces = []

        if element_type == "tetra":
        # For Tetrahedral cells, which have 4 triangular faces
            faces = [
                [element[0], element[1], element[2]],
                [element[0], element[1], element[3]],
                [element[0], element[2], element[3]],
                [element[1], element[2], element[3]]
            ]
        
        elif element_type == "hexahedron":
        # For Hexahedral (cuboid) cells, which have 6 quadrilateral faces
            faces = [
                [element[0], element[1], element[2], element[3]],
                [element[4], element[5], element[6], element[7]],
                [element[0], element[1], element[5], element[4]],
                [element[1], element[2], element[6], element[5]],
                [element[2], element[3], element[7], element[6]],
                [element[3], element[0], element[4], element[7]]
            ]

        elif element_type == "wedge":
            # For Wedge (prism) shaped cells, which have 5 faces, including 3
            # quadrilaterals and 2 triangles)
            faces = [
                [element[0], element[1], element[2]],  # Tri
                [element[3], element[4], element[5]],  # Tri
                [element[0], element[1], element[4], element[3]],  # Quad
                [element[1], element[2], element[5], element[4]],  # Quad
                [element[2], element[0], element[3], element[5]]   # Quad
            ]

        elif element_type == "pyramid":
            # Pyramid has 5 faces (1 quadrilateral, 4 triangles)
            faces = [
                [element[0], element[1], element[2], element[3]],  # Quad
                [element[0], element[1], element[4]],  # Tri
                [element[1], element[2], element[4]],  # Tri
                [element[2], element[3], element[4]],  # Tri
                [element[3], element[0], element[4]]   # Tri
            ]
        # Expect some 2D faces may appear on the boundaries of 3D meshes
        elif element_type == "triangle":
            faces = [element]
        elif element_type == "quad":
            faces = [element]
        
        return faces

    def get_boundary_faces(self):
        """ Function to find the faces of mesh cells that form the boundary of
        that mesh.
         
        Args:
            self : the mesh to find the boundary faces on (this should have
            been read into the MeshReader class already)
           
        Returns:   """
        # List to store boundary faces
        boundary_faces = []

        # Process all cells in mesh
        for cell_block in self.__mesh.cells:
            #print("self__mesh ", self.__mesh)
            element_type = cell_block.type
            # Block type may have numeric ending, removing to make
            # identifying shape easier
            element_type = ''.join(filter(lambda x: x.isalpha(), element_type))

            if element_type in [
                "tetra", "hexahedron", "wedge", "pyramid", "triangle",
                "quad"]:
                # Get faces for each element in the current cell block
                for element in cell_block.data:
                    faces = MeshReader.get_element_faces(element,
                                                         element_type)
                    boundary_faces.extend(faces)

        # Convert to tuples (for easier duplicates handling) and count them
        boundary_faces_tuples = [
            tuple(sorted(face)) for face in boundary_faces]
        face_counts = Counter(boundary_faces_tuples)

        # Boundary faces should appear only once, thus we filter for them
        boundary_faces = [list(face) for face,
                          count in face_counts.items() if count == 1]

        return boundary_faces    

    def _sensor_on_surface(self, model, sensor_idx: list):
        """Function to ensure the sensor lies on the surface of the mesh and
        cannot be free-floating inside the mesh. This is done using
        barycentric coordinates)."""
        bounding_faces = MeshReader.get_boundary_faces(self)
        num_faces = len(bounding_faces)
        
        # use shuffle to randomise sensor-face pairings
        shuffle(bounding_faces)
        # leave order of sensor_idx so that sensor_spots will match
        sensor_spots = []
        for sensor, face in zip(sensor_idx, bounding_faces):
            sensor_spots.append(face)
 
        vertex_dict = {}
        for j in range(0,len(sensor_spots)):
            # get all of the vertices on the face
            vertices = self.mesh.get_face_vertices(sensor_spots[j])
            # 
            vertex_dict = {f"{j}v{i+1}": vertex for i, vertex in enumerate(vertices)}
            vertex_dict = {f"{j}p{i+1}": self.mesh.get_node(vertex) for i, vertex
                           in enumerate(vertices)}

        #v1, v2, v3 = self.mesh.get_face_vertices(sensor_spots)
        #p1, p2, p3 = self.mesh.get_node(v1), self.mesh.get_node(v2), self.mesh.get_node(v3)
        
        x_pos = model.x[sensor_idx]
        y_pos = model.y[sensor_idx]
        z_pos = model.z[sensor_idx]
        
        # Barycentric position calculation
        return (
            model.lambdas[sensor_idx, v1] * p1[0] + model.lambdas[sensor_idx, v2] * p2[0] + model.lambdas[sensor_idx, v3] * p3[0] == x_pos
        ) & (
            model.lambdas[sensor_idx, v1] * p1[1] + model.lambdas[sensor_idx, v2] * p2[1] + model.lambdas[sensor_idx, v3] * p3[1] == y_pos
        ) & (
            model.lambdas[sensor_idx, v1] * p1[2] + model.lambdas[sensor_idx, v2] * p2[2] + model.lambdas[sensor_idx, v3] * p3[2] == z_pos
        )
    
    def get_node(self, index):
        """Get the xyz coordinates of a node by index."""
        return self.nodes[index]

    def get_face_vertices(self, face_index):
        """Get the vertices (node indices) of a face by index."""
        return self.faces[face_index] 