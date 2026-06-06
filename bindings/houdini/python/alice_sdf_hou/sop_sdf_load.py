"""
ALICE-SDF Houdini Python SOP — .asdf Loader
============================================

Python SOP の Code フィールドに以下を貼り付けると、parm `file` で指定した .asdf を
Mesh 化して SOP 出力する。

Parameters:
    file (string)        — .asdf / .asdf.json file path
    bounds_min (float)   — grid min (default -2.0)
    bounds_max (float)   — grid max (default 2.0)
    resolution (int)     — voxel resolution per axis (default 64)
"""

# ==== Houdini Python SOP body (start) ====================================
import hou  # noqa: F401  (provided by Houdini)
import alice_sdf_hou

node = hou.pwd()
geo = node.geometry()
geo.clear()

if not alice_sdf_hou.is_available():
    raise hou.NodeError("alice_sdf backend not installed (cargo build --features python)")

file_path = node.evalParm("file")
bmin = node.evalParm("bounds_min") if node.parm("bounds_min") else -2.0
bmax = node.evalParm("bounds_max") if node.parm("bounds_max") else 2.0
resolution = int(node.evalParm("resolution") if node.parm("resolution") else 64)

sdf = alice_sdf_hou.load_asdf(file_path)
alice_sdf_hou.sdf_to_hou_geo(sdf, geo, bounds=(bmin, bmax), resolution=resolution)
# ==== Houdini Python SOP body (end) ======================================
