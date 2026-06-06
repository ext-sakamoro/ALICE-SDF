"""
ALICE-SDF Houdini Python SOP — SDF Primitive Generator
========================================================

Python SOP の Code に貼り付けて、parameter で sphere / box / torus を選択して生成。

Parameters:
    shape (string menu) — "sphere" | "box" | "torus"
    radius (float)       — sphere / torus major radius
    minor_radius (float) — torus minor
    size_x, size_y, size_z (float) — box
    resolution (int)     — mesh resolution
"""

# ==== Houdini Python SOP body (start) ====================================
import hou  # noqa: F401
import alice_sdf_hou

node = hou.pwd()
geo = node.geometry()
geo.clear()

if not alice_sdf_hou.is_available():
    raise hou.NodeError("alice_sdf backend not installed")

shape = node.evalParm("shape") if node.parm("shape") else "sphere"
resolution = int(node.evalParm("resolution") if node.parm("resolution") else 48)

if shape == "sphere":
    r = node.evalParm("radius") if node.parm("radius") else 1.0
    sdf = alice_sdf_hou.sphere(r)
    bounds = (-r * 1.5, r * 1.5)
elif shape == "box":
    sx = node.evalParm("size_x") if node.parm("size_x") else 1.0
    sy = node.evalParm("size_y") if node.parm("size_y") else 1.0
    sz = node.evalParm("size_z") if node.parm("size_z") else 1.0
    sdf = alice_sdf_hou.box(sx, sy, sz)
    bmax = max(sx, sy, sz) * 1.5
    bounds = (-bmax, bmax)
elif shape == "torus":
    R = node.evalParm("radius") if node.parm("radius") else 1.0
    r = node.evalParm("minor_radius") if node.parm("minor_radius") else 0.3
    sdf = alice_sdf_hou.torus(R, r)
    bmax = (R + r) * 1.3
    bounds = (-bmax, bmax)
else:
    raise hou.NodeError(f"Unknown shape: {shape}")

alice_sdf_hou.sdf_to_hou_geo(sdf, geo, bounds=bounds, resolution=resolution)
# ==== Houdini Python SOP body (end) ======================================
