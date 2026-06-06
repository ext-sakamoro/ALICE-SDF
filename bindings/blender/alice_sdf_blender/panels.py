"""ALICE-SDF Blender side panel (N-panel under "ALICE-SDF" tab)."""

import bpy
from bpy.types import Panel


class ALICESDF_PT_main(Panel):
    bl_label = "ALICE-SDF"
    bl_idname = "ALICESDF_PT_main"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "ALICE-SDF"

    def draw(self, context):
        layout = self.layout
        col = layout.column(align=True)
        col.label(text="Add SDF Primitive:", icon="MESH_DATA")
        col.operator("alicesdf.add_sphere", icon="MESH_UVSPHERE")
        col.operator("alicesdf.add_box", icon="MESH_CUBE")
        col.operator("alicesdf.add_torus", icon="MESH_TORUS")
        layout.separator()
        col2 = layout.column(align=True)
        col2.label(text="Import:", icon="IMPORT")
        col2.operator("alicesdf.import_asdf")


_CLASSES = (ALICESDF_PT_main,)


def register():
    for c in _CLASSES:
        bpy.utils.register_class(c)


def unregister():
    for c in reversed(_CLASSES):
        bpy.utils.unregister_class(c)
