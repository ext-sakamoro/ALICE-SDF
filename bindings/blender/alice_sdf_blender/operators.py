"""ALICE-SDF Blender operators.

Operators:
- ALICESDF_OT_import_asdf: File > Import > ALICE-SDF (.asdf)
- ALICESDF_OT_add_sphere: Add a sphere SDF as Mesh
- ALICESDF_OT_add_box: Add a box SDF as Mesh
- ALICESDF_OT_add_torus: Add a torus SDF as Mesh

すべての操作は alice_sdf Python binding 経由で SDF Tree を構築し、
ALICE-SDF の dual_contouring / marching_cubes で Mesh 化して
Blender Mesh データブロックに流し込む。
"""

import bpy
from bpy.props import FloatProperty, IntProperty, StringProperty
from bpy.types import Operator
from bpy_extras.io_utils import ImportHelper


def _try_import_alice_sdf():
    """Try to import the alice_sdf Python module. Returns module or None."""
    try:
        import alice_sdf  # noqa: F401

        return alice_sdf
    except ImportError:
        return None


def _build_mesh_from_sdf(sdf_module, node, bounds, resolution, mesh_name):
    """SDF node → Blender Mesh data via ALICE-SDF Python binding."""
    # ALICE-SDF Python binding が提供する mesh 生成 API
    # (alice_sdf.sdf_to_mesh が verts + faces を返す想定)
    verts, faces = sdf_module.sdf_to_mesh(node, bounds, resolution)
    mesh = bpy.data.meshes.new(mesh_name)
    mesh.from_pydata(verts, [], faces)
    mesh.update()
    return mesh


def _spawn_mesh(context, mesh, name):
    obj = bpy.data.objects.new(name, mesh)
    context.collection.objects.link(obj)
    context.view_layer.objects.active = obj
    obj.select_set(True)
    return obj


class ALICESDF_OT_import_asdf(Operator, ImportHelper):
    bl_idname = "alicesdf.import_asdf"
    bl_label = "Import ALICE-SDF (.asdf)"
    bl_options = {"REGISTER", "UNDO"}

    filename_ext = ".asdf"
    filter_glob: StringProperty(default="*.asdf;*.asdf.json", options={"HIDDEN"})
    resolution: IntProperty(name="Mesh Resolution", default=64, min=8, max=256)

    def execute(self, context):
        m = _try_import_alice_sdf()
        if m is None:
            self.report({"ERROR"}, "alice_sdf Python module not found (build with --features python)")
            return {"CANCELLED"}
        try:
            node = m.load_asdf(self.filepath)
        except Exception as e:  # noqa: BLE001
            self.report({"ERROR"}, f"Failed to load: {e}")
            return {"CANCELLED"}
        # 既定範囲 = SDF tree の AABB をクエリ
        bounds = (-2.0, 2.0)
        mesh = _build_mesh_from_sdf(m, node, bounds, self.resolution, "AliceSDF_Imported")
        _spawn_mesh(context, mesh, "AliceSDF_Imported")
        return {"FINISHED"}


class ALICESDF_OT_add_sphere(Operator):
    bl_idname = "alicesdf.add_sphere"
    bl_label = "Add ALICE-SDF Sphere"
    bl_options = {"REGISTER", "UNDO"}

    radius: FloatProperty(name="Radius", default=1.0, min=0.01, max=100.0)
    resolution: IntProperty(name="Mesh Resolution", default=48, min=8, max=256)

    def execute(self, context):
        m = _try_import_alice_sdf()
        if m is None:
            self.report({"ERROR"}, "alice_sdf Python module not found")
            return {"CANCELLED"}
        node = m.sphere(self.radius)
        bounds = (-self.radius * 1.5, self.radius * 1.5)
        mesh = _build_mesh_from_sdf(m, node, bounds, self.resolution, "AliceSDF_Sphere")
        _spawn_mesh(context, mesh, "AliceSDF_Sphere")
        return {"FINISHED"}


class ALICESDF_OT_add_box(Operator):
    bl_idname = "alicesdf.add_box"
    bl_label = "Add ALICE-SDF Box"
    bl_options = {"REGISTER", "UNDO"}

    size_x: FloatProperty(name="Size X", default=1.0, min=0.01, max=100.0)
    size_y: FloatProperty(name="Size Y", default=1.0, min=0.01, max=100.0)
    size_z: FloatProperty(name="Size Z", default=1.0, min=0.01, max=100.0)
    resolution: IntProperty(name="Mesh Resolution", default=48, min=8, max=256)

    def execute(self, context):
        m = _try_import_alice_sdf()
        if m is None:
            self.report({"ERROR"}, "alice_sdf Python module not found")
            return {"CANCELLED"}
        node = m.box(self.size_x, self.size_y, self.size_z)
        bmax = max(self.size_x, self.size_y, self.size_z) * 1.5
        bounds = (-bmax, bmax)
        mesh = _build_mesh_from_sdf(m, node, bounds, self.resolution, "AliceSDF_Box")
        _spawn_mesh(context, mesh, "AliceSDF_Box")
        return {"FINISHED"}


class ALICESDF_OT_add_torus(Operator):
    bl_idname = "alicesdf.add_torus"
    bl_label = "Add ALICE-SDF Torus"
    bl_options = {"REGISTER", "UNDO"}

    major_radius: FloatProperty(name="Major Radius", default=1.0, min=0.01, max=100.0)
    minor_radius: FloatProperty(name="Minor Radius", default=0.3, min=0.01, max=100.0)
    resolution: IntProperty(name="Mesh Resolution", default=64, min=8, max=256)

    def execute(self, context):
        m = _try_import_alice_sdf()
        if m is None:
            self.report({"ERROR"}, "alice_sdf Python module not found")
            return {"CANCELLED"}
        node = m.torus(self.major_radius, self.minor_radius)
        bmax = (self.major_radius + self.minor_radius) * 1.3
        bounds = (-bmax, bmax)
        mesh = _build_mesh_from_sdf(m, node, bounds, self.resolution, "AliceSDF_Torus")
        _spawn_mesh(context, mesh, "AliceSDF_Torus")
        return {"FINISHED"}


_CLASSES = (
    ALICESDF_OT_import_asdf,
    ALICESDF_OT_add_sphere,
    ALICESDF_OT_add_box,
    ALICESDF_OT_add_torus,
)


def _menu_import(self, _context):
    self.layout.operator(ALICESDF_OT_import_asdf.bl_idname, text="ALICE-SDF (.asdf)")


def _menu_add(self, _context):
    layout = self.layout
    layout.separator()
    layout.operator(ALICESDF_OT_add_sphere.bl_idname, text="ALICE-SDF Sphere", icon="MESH_UVSPHERE")
    layout.operator(ALICESDF_OT_add_box.bl_idname, text="ALICE-SDF Box", icon="MESH_CUBE")
    layout.operator(ALICESDF_OT_add_torus.bl_idname, text="ALICE-SDF Torus", icon="MESH_TORUS")


def register():
    for c in _CLASSES:
        bpy.utils.register_class(c)
    bpy.types.TOPBAR_MT_file_import.append(_menu_import)
    bpy.types.VIEW3D_MT_add.append(_menu_add)


def unregister():
    bpy.types.VIEW3D_MT_add.remove(_menu_add)
    bpy.types.TOPBAR_MT_file_import.remove(_menu_import)
    for c in reversed(_CLASSES):
        bpy.utils.unregister_class(c)
