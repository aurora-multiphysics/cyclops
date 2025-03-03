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

        return np.array(set_values)
    
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

        grid_pts = np.array(valid_points)

        return grid_pts
    
    def get_node(self, index):
        """Get the xyz coordinates of a node by index."""
        return self.nodes[index]

    def get_face_vertices(self, face_index):
        """Get the vertices (node indices) of a face by index."""
        face_vrts = self.__all_faces[face_index]
        return face_vrts