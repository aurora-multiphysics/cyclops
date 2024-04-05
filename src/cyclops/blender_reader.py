"""
Blender interface for cyclops.

(c) Copyright UKAEA 2024.
"""

import bpy
import bmesh
import numpy as np
import shapely as shp
import matplotlib.pyplot as plt
from itertools import compress
from mathutils.geometry import barycentric_transform, intersect_point_tri_2d
from mathutils import Vector


class MeshObj:
    """Class to read in and manipulate mesh files using Blender functionality.
    Once initialised can output a 2D net of the mesh, test if points are
    contained within this net and track the position of points on the 2D net
    when the net is wrapped around the mesh once more.
    """

    def __init__(self, file_path: str) -> None:
        """Loads a specified mesh file and stores it in a private attribute
        __mesh.

        Args:
        -----
        file_path : (str) the path to and name of the mesh file to be loaded.

        Returns:
        --------
        None
        """
        self.__mesh = bpy.ops.import_mesh.stl(filepath=file_path)
        bpy.ops.object.select_all(action='SELECT')
        bpy.context.active_object.name = 'Mesh_Name'

        OG_obj = bpy.context.object.data

        my_mesh = bmesh.new()
        my_mesh.from_mesh(OG_obj)

        # Get the active mesh
        self.__bmesh = my_mesh
        self.__OG_obj = OG_obj

    def folder(self):
        """Use UV unwrapping to 'unfold' a mesh into a 2D net.

        Args:
        -----
        None

        Returns:
        --------
        uv_points (np.array) an array of points which define the structure
            of the 'UV layer' corresponding to the 3D mesh. This is taken as
            the 2D net for that mesh.

        uv_layer (bpy.types.MeshUVLoopLayer) the blender UV loop of the 3D
            mesh.
        """

        #bpy.ops.object.mode_set(mode='EDIT')
        #bpy.context.scene.objects["Mesh_Name"]
        context = bpy.context
        obj = context.object#edit_object
        me = obj.data
        context.tool_settings.mesh_select_mode = (False, False, True)
        bm = bmesh.new()
        bm.from_mesh(me)
        #bm = bmesh.from_edit_mesh(me)

        #bpy.ops.object.mode_set(mode='EDIT')
        #bpy.ops.mesh.select_all(action='SELECT')
        #bpy.ops.object.mode_set(mode='EDIT', toggle=True)
        bpy.ops.uv.smart_project()
        bpy.ops.object.mode_set(mode='EDIT')

        # old seams
        old_seams = [e for e in bm.edges if e.seam]
        # unmark
        for e in old_seams:
            e.seam = False

        # mark seams from uv islands
        bpy.ops.object.mode_set(mode='EDIT')
        bpy.ops.mesh.select_all(action='SELECT')
        # bpy.ops.uv.select_all({})

        bpy.ops.uv.seams_from_islands()
        seams = bm.edges
        for e in seams:
            e.seam = True

        # split on seams
        #bmesh.types.BMEdgeSeq.ensure_lookup_table()
        bmesh.ops.split_edges(bm, edges=seams)
        # re instate old seams.. could clear new seams.
        for e in old_seams:
            e.seam = True
        bmesh.update_edit_mesh(me)
        #bmesh.ops.split_edges(bm, edges=old_seams)
        #bmesh.update_edit_mesh(me)

        obj = bpy.context.object
        uv_layer = obj.data.uv_layers.active

        bpy.ops.object.mode_set(mode='EDIT')
        bpy.ops.mesh.select_all(action='SELECT')
        bpy.ops.uv.export_layout(mode='SVG', filepath='uv_layout.svg', size=(1024, 1024), opacity=1)
        bpy.ops.uv.smart_project()
        bpy.ops.object.mode_set(mode="EDIT")
        bpy.ops.mesh.select_all(action='TOGGLE')
        # how to set background color of this image?
        bpy.ops.object.mode_set(mode='OBJECT')

        uv_points = []
        # Loops per face, Loop = bpy structure, 1 vertex, 1 edge
        for face in obj.data.polygons:
            for vert_idx, loop_idx in zip(face.vertices, face.loop_indices):
                uv_coords = obj.data.uv_layers.active.data[loop_idx].uv
            #    print(uv_coords)
                uv_points.append((uv_coords.x, uv_coords.y))

        uv_points = np.array(uv_points)
        #print(uv_points)

        return uv_points, uv_layer

    def check_bounds(self, to_check: np.array):
        """Takes an array of points to test and checks if they are contained
        withing the bounds of a given 2D shape.

        Args:
        -----
        to_check : (np.ndarray) n by d array of n points which are being
            tested. Points are made up of float values.

        Returns:
        --------
        np.ndarray : (float) n by d array of n points with d dimensions that
            are contained within the polygon.
        """

        # Get shapely rep of the 2D mesh that makes the shape we are
        # interested in.
        net_coords = test_mesh.folder()[0]
        twoD_shape = shp.polygons(net_coords)

        plt.plot(*twoD_shape.exterior.xy)
        plt.show()
        shp.to_wkt(twoD_shape)
        contained = []
        for i in to_check:
            point = shp.Point(i)
            in_shape = twoD_shape.contains(point)
            contained.append(in_shape)

        passed = list(compress(to_check, contained))
        return passed


def rewrap(uv_vec: np.array, uv_layer):

    my_mesh = test_mesh
    ob = bpy.context.active_object
    bpy.ops.object.mode_set(mode='EDIT')
    me = ob.data
    BM = bmesh.from_edit_mesh(me)

    # Put the UV point on a flat plane
    puv = Vector([uv_vec[0][0], uv_vec[0][1], 0.0])

    bpy.ops.object.mode_set(mode='OBJECT')
    meshdata = bpy.context.active_object.data

    # Loop to iterate over all mesh faces, find where puv is and project onto
    # the 3D mesh
    for i, polygon in enumerate(meshdata.polygons):
        face = []
        co_pts = []
        for i1, loopindex in enumerate(polygon.loop_indices):
            meshloop = meshdata.loops[i1]
            meshvertex = meshdata.vertices[meshloop.vertex_index]
            connected_pt = meshvertex.co.to_3d()
            co_pts.append(connected_pt)

            meshuvloop = meshdata.uv_layers.active.data[loopindex]
            face_point = meshuvloop.uv.to_3d()
            face.append(face_point)
        pa = face[0]
        pb = face[1]
        pc = face[2]
        # Uses Möller–Trumbore intersection algorithm to check if point is on
        # triangle
        if intersect_point_tri_2d(puv, pa, pb, pc):
            pd = co_pts[0]
            pe = co_pts[1]
            pf = co_pts[2]
            pt_transformed = barycentric_transform(puv, pa, pb, pc, pd, pe, pf)

            print('meshuvloop coords: ', meshuvloop.uv, ' selected: ', meshuvloop.select)

    # Get the UV layer
    uv_layer = my_mesh.folder()[1]
    object_methods = [method_name for method_name in dir(uv_layer)]
    print(object_methods)
    # uv_layer = uv_layer.loops.layers.uv['UVMap']
    return pt_transformed


test_object = "/home/cbyers/projects/working_branches/cyclops/tutorials/data/cube.stl"
test_mesh = MeshObj(file_path=test_object)
mesh_net = test_mesh.folder()

points = np.array(((1, 2), (5, -90), (0, 0)))
mesh_bounds = test_mesh.check_bounds(to_check=points)

new_coords = rewrap(uv_vec=mesh_net[0], uv_layer=mesh_net[1])
mest = test_mesh.folder()

# Clearing memory
bpy.ops.wm.read_factory_settings(use_empty=True)
