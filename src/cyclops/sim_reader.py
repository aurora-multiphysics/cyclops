"""
MeshReader classes for cyclops.

Handles reading in simulation data.

(c) Copyright UKAEA 2023.
"""
import numpy as np
import meshio
import pyvista as pv
import warnings

from collections import Counter
from random import shuffle
from scipy.spatial import ConvexHull
from scipy.spatial import Delaunay

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
        self.point_data = self.__mesh.point_data

        if not self.__mesh.point_sets:
            warnings.warn("No named point sets found in mesh file.")
        if self.__mesh.point_sets:
            self.__point_set = self.__mesh.point_sets
        else:
            self.__point_set = {}

        # Extract the nodes (vertices) and faces (dict of faces by type)
        self.__nodes = self.__mesh.points  # Shape (n_nodes, 3) for a 3D mesh
        self.__faces = self.__mesh.cells_dict
        
        # Flatten faces into a single list with their vertex counts
        self.__face_types = list(self.__faces.keys())

        self.__all_faces = []
        for face_type in self.__face_types:
            self.__all_faces.extend(self.__faces[face_type])
        
        # Store the number of faces
        self.__num_faces = len(self.__all_faces)

    def read_pos(self, set_name: str) -> np.ndarray:
        """Record the points described by the region into a numpy array. Mesh
        may be split into 'sets', it is best to read points in by set so that 
        the scalar/vector values for each set are easily matched up.

        Args:
            set_name (str): region name.

        Returns:
            np.ndarray: n by d array of n positions with d dimensions.
        """
        points = []
        for point_index in self.__mesh.point_sets[set_name]:
            points.append(self.__mesh.points[point_index])
        return np.array(points)

    def read_scalar(
        self, set_name: str, scalar_name='all'
    ) -> np.ndarray:
        """Find values of named scalar at the points specified by region name.
        Note that vectors are split into their scalar components and should
        also be read in with this method.

        Args:
            set_name (str): region name
            scalar_name (str): name of the scalar value to read

        Returns:
            np.ndarray: n long numpy array of n scalar values
        """
        set_values = []
        if scalar_name not in self.__mesh.point_data:
            raise KeyError("Scalar '{scalar_name}' not found in mesh"
            "point data.")

        all_values = self.__mesh.point_data[scalar_name]

        if set_name == 'all':
            for region_name in self.read_region_names():
                for point_index in self.__mesh.point_sets[region_name]:
                    set_values.append(all_values[point_index])
        # ToDo possibly add method to look at multiple, but not ALL regions
        else:
            for point_index in self.__mesh.point_sets[set_name]:
                set_values.append(all_values[point_index])

        return np.array(set_values)
    
    def read_region_names(self) -> list:
        """ Obtain the region names within the supplied mesh and return these
          as a list.
          
          Args:
          
          Returns:"""
        region_names = self.__point_set.keys()
        region_blocks = list(region_names)

        return region_blocks

    def num_faces(self) -> int:
        """Return the number of faces in the mesh."""
        return self.__num_faces

    def get_faces(self):
        """Return all faces of the mesh."""
        return self.__all_faces

    def get_element_faces(self, element, element_type):
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

            element_type = cell_block.type
            # Block type may have numeric ending, removing to make
            # identifying shape easier
            element_type = ''.join(filter(lambda x: x.isalpha(), element_type))

            if element_type in [
                "tetra", "hexahedron", "wedge", "pyramid", "triangle",
                "quad"]:
                # Get faces for each element in the current cell block
                for element in cell_block.data:
                    faces = self.get_element_faces(element,
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
        
    def generate_grid(self, resolution: int) -> np.ndarray:
        """Generate a grid of values in the region bounded by a given mesh.

        Args:
            resolution (int): the resolution to generate the grid at, the
            higher the value the finer the grid. (Note that this is not scaled to
            the mesh, the resolution should be adjusted for the size of the mesh in
            questioned)

        Returns:
            np.ndarray: array of grid point positions.
        """
        faces = self.__all_faces
        vertices = self.__mesh.points

        # Flatten into a single 1D list where each set 4 numbers represents a face
        flattened_faces = []

        for face in faces:
            flattened_faces.append(len(face))  # First element: number of vertices
            flattened_faces.extend(face) 

        # Get the bounding box of the mesh
        min_bound = vertices.min(axis=0)
        max_bound = vertices.max(axis=0)

        # Create the 3D grid points using numpy's linspace
        x = np.linspace(min_bound[0], max_bound[0], resolution)
        y = np.linspace(min_bound[1], max_bound[1], resolution)
        z = np.linspace(min_bound[2], max_bound[2], resolution)

        # Generate grid
        grid_x, grid_y, grid_z = np.meshgrid(x, y, z)
        grid_points = np.column_stack(np.meshgrid(x, y, z, indexing="ij")
                                      ).reshape(-1, 3)

        return grid_points

    # Potential problem here, will not work if shape is not convex this will not work
    def clip_grid(self, grid_points):
        """Takes an array of grid points and clips them to keep only those
        that are contained within self.__mesh
        
        Args:
            grid_points (np.ndarray): Grid points to filter.

        Returns:
            np.ndarray: Grid points inside the mesh.

        Warnings:
            This method assumes a convex mesh and may fail for non-convex meshes.
        """
        # Create a ConvexHull object from the vertices of the mesh
        hull = ConvexHull(self.__nodes)

        # Function to check if a point is inside the convex hull
        def point_in_hull(point):
            return Delaunay(hull.points[hull.vertices]).find_simplex(point) >= 0

        # Filter grid points that lie inside mesh using convex hull
        grid_pts = np.array([point for point in grid_points if point_in_hull(point)])

        return grid_pts

    def get_node(self, index):
        """Get the xyz coordinates of a node by index."""
        return self.__nodes[index]

    def get_face_vertices(self, face_index):
        """Get the vertices (node indices) of a face by index."""
        face_vrts = self.__all_faces[face_index]
        return face_vrts
    
    def get_face_vertex_coords(self, face_index):
        """Get the 3D coordinates of vertices of a face by index."""
        face_vrts = self.__all_faces[face_index]  # Indices of the face vertices
        vertex_coords = self.__nodes[face_vrts]   # Convert indices to coordinates
        return vertex_coords

    def compute_face_normal(self, face_index):
        """
        Compute the normal vector of a triangular face.

        Args:
            vertices (np.ndarray): (N, 3) array of vertex positions.
            face (np.ndarray): (3,) array of vertex indices defining the face.

        Returns:
            np.ndarray: Unit normal vector of the face.
        """
        # Get vertex coordinates for the face
        v0, v1, v2 = self.get_face_vertex_coords(face_index)

        # Compute two edge vectors
        edge1 = v1 - v0
        edge2 = v2 - v0

        # Compute the cross product
        normal = np.cross(edge1, edge2)

        # Normalize the normal vector
        norm_length = np.linalg.norm(normal)
        return normal / norm_length if norm_length != 0 else normal

