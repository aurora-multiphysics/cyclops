import bpy
import bmesh
import math
import os
from collections import defaultdict

from time import time
t = time()

test_object = \
    "/home/cbyers/projects/working_branches/cyclops/tutorials/data/cube.stl"
bpy.ops.import_mesh.stl(filepath=test_object)

context = bpy.context
ob = context.object
me = ob.data
bm = bmesh.new()
bm.from_mesh(me)
uvbm = bmesh.new()

uv_layer = bm.loops.layers.uv.verify()
vert_index = uvbm.verts.layers.int.new("index")
face_index = uvbm.faces.layers.int.new("index")
# adjust uv coordinates
for face in bm.faces:
    fverts = []
    for loop in face.loops:
        uv = loop[uv_layer].uv
        v = uvbm.verts.new((uv.x, uv.y, 0))
        v[vert_index] = loop.vert.index
        fverts.append(v)
    f = bmesh.ops.contextual_create(uvbm, geom=fverts)["faces"].pop()
    f[face_index] = face.index

# remove doubles
bmesh.ops.remove_doubles(uvbm, verts=uvbm.verts, dist=1e-7)

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


def get_color(n, e):
    # blender's colors are from 0 to 1
    # so I removed "* 255"
    r = (n[0]*e+1)/2
    g = (n[1]*e+1)/2
    b = (n[2]*e+1)/2
    return (r, g, b, 1)  # added "1", because vcol are rgba


def walk_island(vert):
    ''' walk all un-tagged linked verts '''    
    vert.tag = True
    yield(vert)
    linked_verts = [e.other_vert(vert) for e in vert.link_edges
                    if not e.other_vert(vert).tag]

    for v in linked_verts:
        if v.tag:
            continue
        yield from walk_island(v)


def get_islands(bm, verts=[]):

    def tag(verts, switch):

        for v in verts:
            v.tag = switch

    tag(bm.verts, True)
    tag(verts, False)
    ret = {"islands": []}
    verts = set(verts)
 
    while verts:
        v = verts.pop()
        verts.add(v)
        island = set(walk_island(v))
        print("island: ", island)
        print("faces :", (f.index for x in island for f in
                    x.link_faces))
        faces = set(f.index for x in island for f in
                    x.link_faces if all(v.tag for v in f.verts))
        ret["islands"].append(list(faces))
        tag(island, False)  # remove tag = True
        verts -= island
    return ret


# context = bpy.context
# ob = context.object
# # Loop through the vertices and set their colors
# me = ob.data
# context.tool_settings.mesh_select_mode = (False, False, True)
# e = math.cos(sun.rotation_euler[0])
# # This can be done once (so outside of the loop)
# if len(me.vertex_colors) == 0:
#     me.vertex_colors.new()
# for poly in me.polygons:
#     n = poly.normal
#     # as you need to use the face normal, the calculation below
#     # can be done once per face
#     color = get_color(n, e)
#     print("color ", color)
#     for loop_index in range(poly.loop_start,
#                             poly.loop_start + poly.loop_total):
#         me.vertex_colors.active.data[loop_index].color = color

# me.update()


bm = bmesh.new()
bm.from_mesh(me)
islands = get_islands(bm, verts=bm.verts)["islands"]
off = [False] * len(me.polygons)

# bm.free()
bm.clear()
for island in islands:
    select = off[:]
    print("Looking at island: ", island)
    for i in island:
        print("Current island entry is:", i)
        select[i] = True
    me.polygons.foreach_set("select", select)
    bpy.ops.object.mode_set(mode='EDIT', toggle=True)

bpy.ops.uv.smart_project()
bpy.ops.object.mode_set(toggle=True)

uv_arrange = True
cols, rows = (10, 10)
# if uv_arrange:
    # arrange the UV islands
    # print("Gonna arrange")
bm.from_mesh(me)
output_dir = '/home/cbyers/projects/working_branches/cyclops/'
print(enumerate(island))
for i, island in enumerate(islands):
    print("i is: ", i)
    output_file_pattern_string = 'render%d.jpg'

    # try and move uvs.. prob need to 
    bm.verts.ensure_lookup_table()
    uv_layer = bm.loops.layers.uv.verify()
    bm.faces.ensure_lookup_table()

    # for f in [bm.faces[k] for k in island]:
    #     for l in f.loops:
    #         luv = l[uv_layer]
    #         if luv.uv.x > 1:
    #             luv.uv.x %= 1
    #         luv.uv.x += i % cols
    #         luv.uv.y += i // rows
    bpy.context.scene.render.filepath = os.path.join(
        output_dir, (output_file_pattern_string % i))
    #bpy.ops.render.render(write_still=True)
    bpy.ops.render.render(animation=False, write_still=True)
    # image = bpy.data.images['Render Result']
    # image_name = 'UV_layout%d' + format(i, 'd') + '.png'
    # filename = output_dir + '/' + image_name
    # image.save_render(filename)
bm.to_mesh(me)
me.update()

# bpy.ops.mesh.select_all(action='SELECT')
bpy.ops.uv.export_layout(
    mode='SVG',
    filepath='/home/cbyers/projects/working_branches/cyclops/uv_layout.svg',
    size=(1024, 1024), opacity=1)


print("Finished in ", time() - t, "secs")