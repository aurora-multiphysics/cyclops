import bpy
import bmesh
import os
from collections import defaultdict
from math import radians


# create light datablock, set attributes
light_data = bpy.data.lights.new(name="light_2.80", type='POINT')
light_data.energy = 30

# create new object with our light datablock
light_object = bpy.data.objects.new(name="light_2.80", object_data=light_data)

# link light object
bpy.context.collection.objects.link(light_object)

# make it active 
bpy.context.view_layer.objects.active = light_object

# change location
light_object.location = (5, 5, 5)

# update scene, if needed
dg = bpy.context.evaluated_depsgraph_get() 
dg.update()

test_object = "/home/cbyers/projects/working_branches/cyclops/tutorials/data/cube.stl"
bpy.ops.import_mesh.stl(filepath=test_object)

context = bpy.context
ob = context.object
me = ob.data
bm = bmesh.new()
bm.from_mesh(me)
uvbm = bmesh.new()

uv_layer = bm.loops.layers.uv.verify()
print(uv_layer)
vert_index = uvbm.verts.layers.int.new("index")
print(vert_index)
face_index = uvbm.faces.layers.int.new("index")
print(face_index)

uv_points = []
# Loops per face, Loop = bpy structure, 1 vertex, 1 edge
# for face in obj.data.polygons:
#     for vert_idx, loop_idx in zip(face.vertices, face.loop_indices):
#         uv_coords = obj.data.uv_layers.active.data[loop_idx].uv
#         uv_points.append((uv_coords.x, uv_coords.y))

# adjust uv coordinates
for face in bm.faces:
    print("face ", face.verts)
    fverts = []
    for loop in face.loops:
        print("loop ", loop)
        uv = loop[uv_layer].uv
        print("uv ", uv)
        v = uvbm.verts.new((uv.x, uv.y, 0))
        print("v ", v)
        v[vert_index] = loop.vert.index
        fverts.append(v)
        print("fverts ", fverts)
    f = bmesh.ops.contextual_create(uvbm, geom=fverts)["faces"].pop()
    f[face_index] = face.index

# remove doubles
bmesh.ops.remove_doubles(uvbm, verts=uvbm.verts, dist=1e-7)

'''

# ignore face indices of original if using any option here
# optionally disolve non boundary edges

bmesh.ops.dissolve_edges(uvbm, 
        edges=[e for e in uvbm.edges if not e.is_boundary],
        )


# optionally remove faces

faces = uvbm.faces[:]
while faces:
    uvbm.faces.remove(faces.pop())
'''        
# make an object to see it
me = bpy.data.meshes.new("UVEdgeMesh")
uvbm.to_mesh(me)
ob = bpy.data.objects.new("UVEdgeMesh", me)
bpy.context.collection.objects.link(ob)
ob.show_wire = True

# make a LUT based on verts of original
edge_pairs = defaultdict(list)
boundary_edges = [e for e in uvbm.edges if e.is_boundary]

for e in boundary_edges:
    key = tuple(sorted(v[vert_index] for v in e.verts))
    edge_pairs[key].append(e)

# print result, add text object to show matching edges
# Make sure to remove code below before running on detailed UV as in question, 
# adding that many text objects via operator 
# will slow code down considerably.
uvbm.verts.ensure_lookup_table()
for key, edges in edge_pairs.items():

    print(key, [e.index for e in edges]) 
    for e in edges:
        if not e.is_boundary:
            continue
        f = e.link_faces[0]
        p = (e.verts[0].co + e.verts[1].co) / 2
        p += (f.calc_center_median() - p) / 4
        bpy.ops.object.text_add(radius=0.04, location=p)
        bpy.context.object.data.body = f"{key}"

jpg_path = os.path.join("/home/cbyers/projects/working_branches/cyclops/", 'uv_layout' + '.jpg')
#os.modifiers["GeometryNodes"]["Input_12"] = jpg_path #update filename in image
            
# output to jpg file
scene = bpy.context.scene
scene.render.image_settings.file_format='JPEG'
scene.render.filepath=jpg_path
bpy.ops.render.render(write_still=1)


# def rotate_and_render(output_dir,
#                       output_file_pattern_string='render%d.jpg',
#                       rotation_steps=32,
#                       rotation_angle=360.0,
#                       subject=bpy.context.object):

#     original_rotation = subject.rotation_euler
#     for step in range(0, rotation_steps):
#         subject.rotation_euler[2] = radians(
#             step * (rotation_angle/rotation_steps))
#         bpy.context.scene.render.filepath = os.path.join(
#             output_dir, (output_file_pattern_string % step))
#         bpy.ops.render.render(write_still=True)
#     subject.rotation_euler = original_rotation


# rotate_and_render("/home/cbyers/projects/working_branches/cyclops/",
#                   'render%d.jpg', subject=ob)
